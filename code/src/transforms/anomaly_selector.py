import torch
from . import MultiNodeData, NodeSelector, anomaly_names
import logging


class AnomalySelector(object):
    "Select only a subset of anomalies."
    
    def __init__(self, exclude_anomalies: list = []):
        self.__indices__ = None
        self.__exclude_normal__ = False
        self.exclude_anomalies = exclude_anomalies

        if len(self.exclude_anomalies):
            self.__indices__ = torch.tensor([idx for idx, el in enumerate(
                anomaly_names) if el not in self.exclude_anomalies or el == "normal"], dtype=torch.long)
            self.__exclude_normal__ = "normal" in self.exclude_anomalies

    def __repr__(self):
        return f"{self.__class__.__name__}(exclude_anomalies={self.exclude_anomalies})"

    def __str__(self):
        return f"{self.__class__.__name__}(exclude_anomalies={self.exclude_anomalies})"

    def __get_clean_object__(self):
        info_dict: dict = {"skip_me": True}
        return MultiNodeData(info_dict)

    def __call__(self, data: MultiNodeData):

        identifiers: list = data["identifiers"]
        # create detailed label tensor
        class_labels = [data[f"y_{identifier}"] for identifier in identifiers]
        data["y_full"] = torch.cat(class_labels, dim=0)

        # exclude anomalies if specified
        if self.__indices__ is not None:
            # override detailed batched label tensor
            y_full = data["y_full"][:, self.__indices__]

            # override individual label tensors
            for identifier in identifiers:
                data[f"y_{identifier}"] = data[f"y_{identifier}"][:, self.__indices__]

            # we have to check if we also need to remove corresponding node data
            # returns (values, indices), we only need "values"
            row_max = torch.max(y_full, dim=1)[0]
            if y_full.size(0) != torch.sum(row_max):
                new_indices = [ind for (ind, max_val) in zip(
                    range(y_full.size(0)), torch.flatten(row_max)) if max_val != 0]
                new_identifiers = [identifiers[ind] for ind in new_indices]

                # clean y_full
                data["y_full"] = y_full[new_indices, :]
                # we still need to clean the data-object by removing unneeded nodes
                node_selector = NodeSelector(include_nodes=new_identifiers)
                data = node_selector(data)
            else:
                data["y_full"] = y_full

        y_full = data["y_full"]
        # sanity check
        if torch.sum(y_full[:, 1:]) > 1:
            error_message = "Multiple Anomalies at same time! Not allowed."
            logging.error(error_message)
            raise ValueError(error_message)

        # create compact label tensor
        y_compact = torch.sum(y_full, dim=0).reshape(1, -1)
        y_compact = (y_compact > 0).type_as(y_compact)
        if torch.sum(y_compact) > 1:
            # if there is at least one anomaly, this graph cant be "normal"
            y_compact[0, 0] = 0

        data["y"] = y_compact

        # if we want to exclude "normal" graph-samples
        if self.__exclude_normal__:
            if y_compact[0, 0] == 1:
                data = self.__get_clean_object__()
            else:
                data["y"] = y_compact[:, 1:]    

        return data
