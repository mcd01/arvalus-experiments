import torch
from . import MultiNodeData


__full_anomaly_dict__ = {
    "normal": 0,
    "bandwidth": 1,
    "download": 2,
    "packet_duplication": 3,
    "packet_loss": 4,
    "stress_hdd": 5,
    "stess_hdd": 5,
    "mem_leak": 6,
    "stress_cpu": 7,
    "stress_mem": 8
}

__compact_anomaly_dict__ = {
    "normal": 0,
    "bandwidth": 1,
    "download": 2,
    "packet_duplication": 3,
    "packet_loss": 4,
    "stress_hdd": 5,
    "mem_leak": 6,
    "stress_cpu": 7,
    "stress_mem": 8
}

anomaly_codes = sorted(list(set(list(__compact_anomaly_dict__.values()))))
anomaly_names = [dict(map(reversed, __compact_anomaly_dict__.items()))[code] for code in anomaly_codes]


class DataLabeler(object):
    "Assign labels to extracted sub-graphs. Based on tag-information."
    
    def __init__(self, min_occurrence : float = 0.5):
        self.min_occurrence = min_occurrence
        self.__count_dict__ : dict = {}
    
    def __repr__(self):
        return f"{self.__class__.__name__}(min_occurrence={self.min_occurrence})"
    
    def __str__(self):
        return f"{self.__class__.__name__}(min_occurrence={self.min_occurrence})"
    
    def __get_class_information__(self, anomaly_name: str):
        global __full_anomaly_dict__
        global anomaly_codes
        
        # for an anomaly name, get the corresponding scalar 
        if anomaly_name in __full_anomaly_dict__:
            return anomaly_name, __full_anomaly_dict__[anomaly_name], anomaly_codes
        else:
            r_key, r_anomaly = None, None
            for key in __full_anomaly_dict__.keys():
                if key in anomaly_name:
                    r_key, r_anomaly = key, __full_anomaly_dict__[key]
                    break
            return r_key, r_anomaly, anomaly_codes

    def __get_anomaly_name__(self, label_list : str):
        count_dict : dict = {}
        # count label occurrences
        for label in label_list:
            if label in count_dict:
                count_dict[label] += 1
            else:
                count_dict[label] = 1
        # get key of label with highest occurrence
        max_key : str = max(count_dict, key=count_dict.get)
        # decide for a final label, incorporate defined min_occurrence
        anomaly_name :str = max_key if count_dict[max_key] >= int(len(label_list) * self.min_occurrence) else "normal"
        
        return anomaly_name
    
    def __call__(self, data: MultiNodeData):
        identifiers : list = data["identifiers"]
        
        for identifier in identifiers:
            label_list : list = data[f"label_list_{identifier}"]
            label_list = sorted(label_list)
            label_list_key : str = str(label_list)
            
            anomaly_name : str = None
            if label_list_key in self.__count_dict__: # if cached, use cached version
                anomaly_name = self.__count_dict__[label_list_key]
            else: # otherwise, "compute" from tag-list
                anomaly_name = self.__get_anomaly_name__(label_list)
                self.__count_dict__[label_list_key] = anomaly_name
            
            # get name and code of anomaly (+ raw code list)
            anomaly_name, anomaly_code, anomaly_codes = self.__get_class_information__(anomaly_name)
            
            # enhance object
            data[f"y_{identifier}"] = torch.tensor([int(anomaly_code == el) for el in anomaly_codes], dtype=torch.long).reshape(1, -1) # encode multi label
            data[f"anomaly_{identifier}"] = anomaly_name
            # cleanup
            if hasattr(data, f"label_list_{identifier}"):
                delattr(data, f"label_list_{identifier}")
            
        return data