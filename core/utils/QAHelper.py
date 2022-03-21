import numpy as np
from core.utils.report import ReportAnnotation
from core.utils.parser import AnnotationParser
from core.utils.misc import encode_list, decode_list

class QAHelper():
    def __init__(self, target_video, gt_json_path, inference_json_path, extract_interval, output_path):
        self.target_video = target_video
        self.gt_json_path = gt_json_path
        self.inference_json_path = inference_json_path
        self.extract_interval = extract_interval
        self.output_path = output_path

        self.CORRECT_CLS = 100
        self.OVER_GT_CLS = 0
        self.OVER_INFER_CLS = 1
        self.ONLY_INFER_CLS = 2
        self.RS_CLS, self.NRS_CLS = (0,1)

    
    def compare_to_json(self):

        # prepare
        gt_anno=AnnotationParser(self.gt_json_path)
        inference_anno=AnnotationParser(self.inference_json_path)

        gt_event_seq = gt_anno.get_event_sequence(self.extract_interval)
        inference_event_seq = inference_anno.get_event_sequence(self.extract_interval)

        # compare
        gt_event_seq = np.array(gt_event_seq)
        inference_event_seq = np.array(inference_event_seq)

        # adjust min length
        adj_len = min(len(gt_event_seq), len(inference_event_seq))
        gt_event_seq, inference_event_seq= gt_event_seq[:adj_len], inference_event_seq[:adj_len]
        
        sub_seq = gt_event_seq - inference_event_seq # [1,0,0,-1,-1,0,1, ...]
        
        # change cls
        sub_seq_enco = encode_list(sub_seq.tolist())
        results_seq_enco = []

        start_idx = 0
        end_idx = 0
        for value, enco_cls in sub_seq_enco:
            end_idx = start_idx + value
            
            if enco_cls == 0:
                results_seq_enco.append([value, self.CORRECT_CLS])
            elif enco_cls == 1:
                results_seq_enco.append([value, self.OVER_GT_CLS])
            elif enco_cls == -1: # => -1 or 2
                # check only, over infer class
                is_only_infer_case = False

                pre_idx = 0 if start_idx - 1 < 0 else start_idx - 1 # exception for overindexing
                pro_idx = adj_len - 1 if end_idx + 1 >= adj_len else end_idx # exception for overindexing
                
                pre_gt_cls, pro_gt_cls = gt_event_seq[pre_idx], gt_event_seq[pro_idx]
                pre_inf_cls, pro_inf_cls = inference_event_seq[pre_idx], inference_event_seq[pro_idx]
                
                if (pre_gt_cls == self.RS_CLS) and (pre_inf_cls == self.NRS_CLS): # FP
                    is_only_infer_case = True
                
                if (pro_gt_cls == self.RS_CLS) and (pro_inf_cls == self.NRS_CLS): # FP
                    is_only_infer_case = True
                    
                if is_only_infer_case: # 2
                    results_seq_enco.append([value, self.ONLY_INFER_CLS])
                else: # -1
                    results_seq_enco.append([value, self.OVER_INFER_CLS])                
            
            start_idx = end_idx
        
        results_seq = decode_list(results_seq_enco) # [0,1,1,0,0,1,1,1,..]

        # save to json
        # meta info
        totalFrame = gt_anno.get_totalFrame()
        frameRate = gt_anno.get_fps()
        width="temp"
        height="temp"
        name="temp"
        createdAt = "temp"
        updatedAt = "temp"
        _id = "temp"
        annotationType = "NRS"
        annotator = "temp"
        label = {str(self.CORRECT_CLS): "CORRECT",
                    str(self.OVER_GT_CLS): "OVER_GT",
                    str(self.OVER_INFER_CLS): "OVER_INFERENCE",
                    str(self.ONLY_INFER_CLS): "ONLY_INFERENCE"}

        compare_report = ReportAnnotation(self.output_path)
        compare_report.set_total_report(totalFrame, frameRate, width, height, _id, annotationType, createdAt, updatedAt, annotator, name, label)

        # add annotation
        results_chunk_cnt = len(results_seq_enco)
        start_frame = 0
        end_frame = 0
        for i, (value, enco_cls) in enumerate(results_seq_enco):
            end_frame = start_frame + (value * self.extract_interval) - 1
            
            if enco_cls == self.CORRECT_CLS:
                pass

            else:
                # check over totalFrame on last annotation (because of quntization? when set up extract_interval > 1)
                if i == results_chunk_cnt and end_frame >= totalFrame: 
                    end_frame = totalFrame - 1

                compare_report.add_annotation_report(start_frame, end_frame, code=enco_cls)


            start_frame = end_frame + 1

        compare_report.save_report()



        
        


        


            

        
        




    

        

        

