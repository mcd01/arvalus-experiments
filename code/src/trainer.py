import torch
from src.utils import create_supervised_trainer, create_supervised_evaluator, create_dirs
from src.metrics import LabelwiseAccuracy, CpuInfo
from ignite.metrics import Accuracy, Loss, Recall, Precision, Fbeta, ConfusionMatrix
from ignite.engine import Events
import json
import os
import glob
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine, EarlyStopping, TerminateOnNan
from datetime import datetime
from collections import OrderedDict
import pandas as pd
import sys
import torch_geometric


# score function to be evaluated for early stopping
def score_function(engine):
    val_loss = engine.state.metrics['loss']
    return -val_loss

def metric_output_transform(output, loss_wrapper, target : str, collapse_y : bool = False):
    """Prepare model output before computing metrics."""
    
    y_pred, y_true = output
    y_pred_fine, y_pred_coarse = y_pred
    y_true_fine, y_true_coarse = y_true
    
    new_y_pred = y_pred_fine if target == "node" else y_pred_coarse
    new_y_true = y_true_fine if target == "node" else y_true_coarse
    
    temp_y_pred = torch.zeros_like(new_y_true)
    indices = torch.argmax(new_y_pred, dim=1)
    temp_y_pred.scatter_(1, indices.view(-1, 1), 1)
    new_y_pred = temp_y_pred
        
    if collapse_y:
        new_y_true = torch.argmax(new_y_true, dim=1)        
        
    return new_y_pred, new_y_true
      
    
class Trainer():
    """Wrapper for training and inference process."""
    
    def __init__(self, trainer_args={}):

        self.log_prefix = trainer_args.get("model_name")
            
        self.save_path = os.path.join(trainer_args.get("fold_dir"))

        create_dirs(self.save_path)

        self.device = trainer_args.get("device")
        self.epochs = trainer_args.get("epochs")
        self.early_stopping = trainer_args.get("early_stopping", None)

        self.exclude_anomalies = trainer_args.get("exclude_anomalies")
        self.include_metrics = trainer_args.get("include_metrics")

        self.model_class = trainer_args.get("model_class")
        self.model_args = trainer_args.get("model_args", {})

        self.optimizer_class = trainer_args.get("optimizer_class")
        self.optimizer_args = trainer_args.get("optimizer_args", {})

        self.training_data_stats = trainer_args.get("training_data_stats")
        
        self.loss = trainer_args.get("loss_func")
        self.node_classes = trainer_args.get("node_classes")
        self.graph_classes = trainer_args.get("graph_classes")

        self.score_function = score_function

        self.resume_from_checkpoint = trainer_args.get("resume_from_checkpoint", {})
        
        # create model, optimizer
        self.model = self.model_class({
            **self.model_args, 
            "pred_collector_function": self._pred_collector_function
        }).to(self.device).double()
        
        self.optimizer = self.optimizer_class(
            self.model.parameters(), **self.optimizer_args)
        self.trainer = create_supervised_trainer(
            self.model, 
            self.optimizer, 
            loss_fn=self.loss, 
            device=self.device, 
            non_blocking=True,
            output_transform=lambda x, y, y_pred, loss: (y_pred, y) # so that loss-metric can work with transformed output
            )
        
        ##################### log some values ################################
        if not len(trainer_args.get("resume_from_checkpoint", {})): # only print all this on initial training start, not on resume
            self.custom_print("Device:", self.device)
            self.custom_print("Max. epochs:", self.epochs)
            self.custom_print("Early stopping:", self.early_stopping)
            self.custom_print("Excluded Anomalies:", self.exclude_anomalies)
            self.custom_print("Included Metrics:", self.include_metrics)
            self.custom_print("Loss class:", self.loss)
            self.custom_print("Node anomaly classes:", self.node_classes)
            self.custom_print("Graph anomaly classes:", self.graph_classes)
            self.custom_print("Model class:", self.model_class)
            self.custom_print("Model args:", json.dumps(self.model_args))
            self.custom_print("Model - All Parameters:", self.model.all_params)
            self.custom_print("Model - Trainable Parameters:", self.model.all_trainable_params)
            self.custom_print("Optimizer args:", json.dumps(self.optimizer_args))
            self.custom_print("Training Data Statistics:", self.training_data_stats)
            self.custom_print("Train indices", trainer_args.get("train_indices"))
            self.custom_print("Val indices", trainer_args.get("val_indices"))
            self.custom_print("Test indices", trainer_args.get("test_indices"))
        ######################################################################
        
        # configure behavior for early stopping
        self.stopper = None
        if self.early_stopping:
            self.stopper = EarlyStopping(
                patience=self.early_stopping, score_function=self.score_function, trainer=self.trainer)
            
        
        # configure behavior for checkpointing
        to_save: dict = {
            "model_state_dict": self.model,
            "optimizer_state_dict": self.optimizer,
            "trainer_state_dict": self.trainer
        }    
        if self.stopper:
            to_save["stopper_state_dict"] = self.stopper
            
        save_handler = DiskSaver(self.save_path, create_dir=True,
                                    require_empty=False, atomic=True)   
        
        # save the best checkpoints
        self.best_checkpoint_handler = Checkpoint(
                to_save,
                save_handler,
                filename_prefix=f"{self.log_prefix}_best",
                score_name="val_loss",
                score_function=self.score_function,
                include_self=True,
                global_step_transform=global_step_from_engine(self.trainer),
                n_saved=5) 
        
        # save the latest checkpoint (important for resuming training)
        self.latest_checkpoint_handler = Checkpoint(
                to_save,
                save_handler,
                filename_prefix=f"{self.log_prefix}_latest",
                include_self=True,
                global_step_transform=global_step_from_engine(self.trainer),
                n_saved=1)
            

        # resume from checkpoint
        if len(self.resume_from_checkpoint):
            self.model, self.optimizer, self.trainer, self.stopper, self.best_checkpoint_handler, self.latest_checkpoint_handler = self._load_checkpoint(
                self.model, self.optimizer, self.trainer, self.stopper, self.best_checkpoint_handler, self.latest_checkpoint_handler, checkpoint_path_dict=self.resume_from_checkpoint)

        self.persist_collection = False
        self.persist_collection_dict : OrderedDict = OrderedDict()

    def _pred_collector_function(self, key, value):
        """Collect predictions on (validation/test)-set."""

        if self.persist_collection:
            if key not in self.persist_collection_dict:
                self.persist_collection_dict[key] = []
            self.persist_collection_dict[key].append(value)

    def _save_collected_predictions(self, prefix="test"):
        """Saves intermediate results to csv."""
        
        self.persist_collection = False
        
        red_dict : OrderedDict = OrderedDict()
            
        for key, value in self.persist_collection_dict.items():
            if all([isinstance(el, torch.Tensor) for el in value]):
                red = torch.cat(value, dim=0).tolist()
            elif all([isinstance(el, list) for el in value]):
                red = sum(value, [])
            else:
                red = value
            
            red_dict[key] = red
                
        pd.DataFrame.from_dict(red_dict).to_csv(os.path.join(self.save_path, f"{self.log_prefix}_{prefix}_results.csv"), index=False)
        
        self.persist_collection_dict : OrderedDict = OrderedDict()

    def _load_checkpoint(self, model, optimizer, trainer, stopper, best_checkpoint_handler, latest_checkpoint_handler, checkpoint_path_dict: dict = {}):
        """Load a model from checkpoint(s)."""
        
        best_checkpoint_path = checkpoint_path_dict.get("best_checkpoint_path", None)
        latest_checkpoint_path = checkpoint_path_dict.get("latest_checkpoint_path", None)
        
        if latest_checkpoint_path:
            latest_checkpoint = torch.load(latest_checkpoint_path)
            
            model.load_state_dict(latest_checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(latest_checkpoint['optimizer_state_dict'])
            trainer.load_state_dict(latest_checkpoint['trainer_state_dict'])
            
            if stopper and "stopper_state_dict" in latest_checkpoint:
                stopper.load_state_dict(latest_checkpoint['stopper_state_dict'])
                
            if latest_checkpoint_handler and "checkpointer" in latest_checkpoint:    
                latest_checkpoint_handler.load_state_dict(latest_checkpoint["checkpointer"])    
                
        if best_checkpoint_path:
            best_checkpoint = torch.load(best_checkpoint_path)        
            if best_checkpoint_handler and "checkpointer" in best_checkpoint:
                best_checkpoint_handler.load_state_dict(best_checkpoint["checkpointer"])
        
        
        self.custom_print(f"Resume training from epoch={trainer.state.epoch} with max_epoch={self.epochs} and early_stopping={self.early_stopping}...")
        return model, optimizer, trainer, stopper, best_checkpoint_handler, latest_checkpoint_handler

    def custom_print(self, *args):
        """Redirect print-output to additional file."""

        text = ' '.join(str(i) for i in args)
        if isinstance(text, dict):
            text = json.dumps(text)
        
        text = f"{datetime.now()} - {text}" 

        print(text)
        with open(os.path.join(self.save_path, '{}.txt'.format(self.log_prefix)), mode='a') as file_object:
            print(text, file=file_object)
            
    def write_metrics(self, train_engine, ref_engine, suffix="validation"):
        """Write metrics to file."""
        
        def list_to_string(*args, **kwargs):
            tmp_list = []
            for a in args:
                tmp_el = None
                if isinstance(a, dict):
                    tmp_el = json.dumps(a)
                elif isinstance(a, torch.Tensor):
                    tmp_el = a.tolist()
                else:
                    tmp_el = a
                
                tmp_list.append(str(tmp_el))
                
            return "|".join(tmp_list)    
        
        epoch = train_engine.state.epoch if suffix == "validation" else -1
        train_time = train_engine.state.times
        ref_time = ref_engine.state.times
        
        train_metrics = OrderedDict(sorted(train_engine.state.metrics.items(), key=lambda m: m[0]))
        ref_metrics = OrderedDict(sorted(ref_engine.state.metrics.items(), key=lambda m: m[0]))
        
        with open(os.path.join(self.save_path, '{}_{}.csv'.format(self.log_prefix, suffix)), mode='a') as file_object:
            # print header line if required
            if epoch == 1 or suffix != "validation":
                header_line = ["epoch"] + [f"train_{k}" for k in ["time"] + list(train_metrics.keys())] + [f"{suffix}_{k}" for k in ["time"] + list(ref_metrics.keys())]
                print(list_to_string(*header_line), file=file_object)
            # print actual values
            value_line = [epoch, train_time] + list(train_metrics.values()) + [ref_time] + list(ref_metrics.values())
            print(list_to_string(*value_line), file=file_object) 

    def run(self, train_loader, val_loader, test_loader):
        """Perform model training and evaluation on holdout dataset."""
        
        ## attach certain metrics to trainer ##
        CpuInfo().attach(self.trainer, "cpu_util")
        Loss(self.loss).attach(self.trainer, "loss")

        ###### configure evaluator settings ######
        def get_output_transform(target:str, collapse_y:bool=False):
            return lambda out: metric_output_transform(out, self.loss, target, collapse_y=collapse_y)
        
        graph_num_classes = len(self.graph_classes)
        node_num_classes = len(self.node_classes)
        node_num_classes = 2 if node_num_classes == 1 else node_num_classes
        
        node_output_transform = get_output_transform("node")
        node_output_transform_collapsed = get_output_transform("node", collapse_y=True)
        graph_output_transform = get_output_transform("graph")
        graph_output_transform_collapsed = get_output_transform("graph", collapse_y=True)
        
        # metrics we are interested in
        base_metrics : dict = {
            'loss': Loss(self.loss),
            "cpu_util": CpuInfo(),
            'node_accuracy_avg': Accuracy(output_transform=node_output_transform, is_multilabel=False),
            'node_accuracy': LabelwiseAccuracy(output_transform=node_output_transform, is_multilabel=False),
            "node_recall": Recall(output_transform=node_output_transform_collapsed, is_multilabel=False, average=False),
            "node_precision": Precision(output_transform=node_output_transform_collapsed, is_multilabel=False, average=False),
            "node_f1_score": Fbeta(1, output_transform=node_output_transform_collapsed, average=False),
            "node_c_matrix": ConfusionMatrix(node_num_classes, output_transform=node_output_transform_collapsed, average=None)
        }
            
        metrics = dict(**base_metrics)    
        
        # settings for the evaluator
        evaluator_settings = {
            "device": self.device,
            "loss_fn": self.loss,
            "node_classes": self.node_classes,
            "graph_classes": self.graph_classes,
            "non_blocking": True,
            "metrics": OrderedDict(sorted(metrics.items(), key=lambda m: m[0])),
            "pred_collector_function": self._pred_collector_function
        }

        ## configure evaluators ##
        val_evaluator = None
        if len(val_loader):
            val_evaluator = create_supervised_evaluator(self.model, **evaluator_settings)
            # configure behavior for early stopping
            if self.stopper:
                val_evaluator.add_event_handler(Events.COMPLETED, self.stopper)
            # configure behavior for checkpoint saving
            val_evaluator.add_event_handler(Events.COMPLETED, self.best_checkpoint_handler)
            val_evaluator.add_event_handler(Events.COMPLETED, self.latest_checkpoint_handler)
        else:
            self.trainer.add_event_handler(Events.COMPLETED, self.latest_checkpoint_handler)

        test_evaluator = None
        if len(test_loader):
            test_evaluator = create_supervised_evaluator(self.model, **evaluator_settings)
        #############################

        @self.trainer.on(Events.STARTED)
        def log_training_start(trainer):
            self.custom_print("Start training...")               

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def compute_metrics(trainer):
            """Compute evaluation metric values after each epoch."""
            
            epoch = trainer.state.epoch
            
            self.custom_print(f"Finished epoch {epoch:03d}!")

            if len(val_loader):
                self.persist_collection = True
                val_evaluator.run(val_loader)
                self._save_collected_predictions(prefix=f"validation_epoch{epoch:03}")
                # write metrics to file
                self.write_metrics(trainer, val_evaluator, suffix="validation")

        @self.trainer.on(Events.COMPLETED)
        def log_training_complete(trainer):
            """Trigger evaluation on test set if training is completed."""

            epoch = trainer.state.epoch
            suffix = "(Early Stopping)" if epoch < self.epochs else ""

            self.custom_print("Finished after {:03d} epochs! {}".format(
                epoch, suffix))

            # load best model for evaluation
            self.custom_print("Load best model for final evaluation...")
            last_checkpoint : str = self.best_checkpoint_handler.last_checkpoint or self.latest_checkpoint_handler.last_checkpoint
            best_checkpoint_path = os.path.join(self.save_path, last_checkpoint)
            checkpoint_path_dict: dict = {
                "latest_checkpoint_path": best_checkpoint_path # we want to load states from the best checkpoint as "latest" configuration for testing
            }
            self.model, self.optimizer, self.trainer, _, _, _ = self._load_checkpoint(
                self.model, self.optimizer, self.trainer, None, None, None, checkpoint_path_dict=checkpoint_path_dict)

            if len(test_loader):
                self.persist_collection = True
                test_evaluator.run(test_loader)
                self._save_collected_predictions(prefix="test_final")
                # write metrics to file
                self.write_metrics(trainer, test_evaluator, suffix="test")

        # terminate training if Nan values are produced
        self.trainer.add_event_handler(
            Events.ITERATION_COMPLETED, TerminateOnNan())

        # start the actual training
        self.custom_print(f"Train for a maximum of {self.epochs} epochs...")
        self.trainer.run(train_loader, max_epochs=self.epochs)
