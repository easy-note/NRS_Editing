
def extract_target_video_path(target_video_path):
    import os
    '''
    target_video_path = /data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case/L_1/04_GS4_99_L_1_01.mp4
    base_path = /data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case/L_1
    target_path = /data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case/L_1/04_GS4_99_L_1_01
    '''
    
    base_path = '/'.join(target_video_path.split('/')[:-1])
    target_path = os.path.join(base_path, target_video_path.split('/')[-1].split('.')[0]) # 마지막 비디오 명에 '.' 이 들어가 있으면 에러 -> naming rule 지켜져야 함. 

    return target_path


def get_video_meta_info(target_video):
    import os
    import glob
    import datetime

    video_name = target_video.split('/')[-1]
    video_path = target_video
    frame_list = glob.glob(os.path.join(extract_target_video_path(target_video), 'frames', '*.jpg'))
    print(os.path.join(extract_target_video_path(target_video), 'frame', '*.jpg'))
    date_time = str(datetime.datetime.now())

    return video_name, video_path, len(frame_list), date_time


def save_meta_log(target_video, output_base_path):
    import json
    from collections import OrderedDict

    import os
    '''
    	"04_GS4_99_L_1_01.mp4": {
            "video_path": "/data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case/L_1/04_GS4_99_L_1_01.mp4",
            "frame_cnt": 108461,
            "date": "2022-01-06 17:55:36.291518"
	}
    '''

    print('\nmeta log saving ...')

    meta_log_path = os.path.join(output_base_path, 'logs')
    os.makedirs(meta_log_path, exist_ok=True)
    
    meta_data = OrderedDict()
    video_name, video_path, frame_cnt, date_time = get_video_meta_info(target_video)

    meta_data[video_name] = {
        'video_path': video_path,
        'frame_cnt': frame_cnt,
        'date': date_time
    }

    print(json.dumps(meta_data, indent='\t'))

    try: # existing editing_log.json 
        with open(os.path.join(meta_log_path, 'editing_logs', 'editing_log.json'), 'r+') as f:
            data = json.load(f)
            data.update(meta_data)

            f.seek(0)
            json.dump(data, f, indent=2)
    except:
        os.makedirs(os.path.join(meta_log_path, 'editing_logs'), exist_ok=True)

        with open(os.path.join(meta_log_path, 'editing_logs', 'editing_log.json'), 'w') as f:
            json.dump(meta_data, f, indent=2)




def frame_cutting(target_video):
    import sys
    import os    

    print('\nframe cutting ... ')

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    
    sys.path.append(base_path)
    sys.path.append(base_path+'core')

    from core.utils.ffmpegHelper import ffmpegHelper

    # save_path 
    frame_save_path = os.path.join(extract_target_video_path(target_video), 'frames')
    os.makedirs(frame_save_path, exist_ok=True)

    # frame cutting -> save to '/data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case/L_1/04_GS4_99_L_1_01/frames/frame-000000000.jpg'
    ffmpeg_helper = ffmpegHelper(target_video, frame_save_path)
    ffmpeg_helper.cut_frame_total()


def get_experiment_args():
    from core.config.base_opts import parse_opts

    parser = parse_opts()

    args = parser.parse_args()

    ### model basic info opts
    args.pretrained = True
    # TODO 원하는대로 변경 하기
    # 전 그냥 save path와 동일하게 가져갔습니다. (bgpark)
    args.save_path = args.save_path + '-trial:{}-fold:{}'.format(args.trial, args.fold)
    # args.save_path = args.save_path + '-model:{}-IB_ratio:{}-WS_ratio:{}-hem_extract_mode:{}-top_ratio:{}-seed:{}'.format(args.model, args.IB_ratio, args.WS_ratio, args.hem_extract_mode, args.top_ratio, args.random_seed) # offline method별 top_ratio별 IB_ratio별 실험을 위해
    args.experiments_sheet_dir = args.save_path

    ### dataset opts
    args.data_base_path = '/raid/img_db'

    ### train args
    args.num_gpus = 1
    
    ### etc opts
    args.use_lightning_style_save = True # TO DO : use_lightning_style_save==False 일 경우 오류해결 (True일 경우 정상작동)

    return args


def inference(target_video, output_path, model_path = '/NRS_EDITING/logs_sota/mobilenetv3_large_100-general-rs/version_1/checkpoints/epoch=19-Mean_metric=0.9903-best.ckpt'):
    '''
    inference
    result_save_path : /data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case/L_1/04_GS4_99_L_1_01/results/04_GS4_99_L_1_01.json
    추후 정량 평가 가능하면, results 내에 결과 저장. (CR, OR 등)
    '''
    import os
    import pandas as pd
    import glob
    import sys

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    
    sys.path.append(base_path)
    sys.path.append(base_path+'core')
    sys.path.append(base_path+'/core/accessory/RepVGG')

    from core.utils.ffmpegHelper import ffmpegHelper

    ### inference_main
    ### test inference module
    from core.api.trainer import CAMIO
    from core.api.inference import InferenceDB # inference module
    from core.api.evaluation import Evaluator # evaluation module
    from core.utils.metric import MetricHelper # metric helper (for calc CR, OR, mCR, mOR)
    from core.utils.logger import Report # report helper (for experiments reuslts and inference results)

    from core.api.visualization import VisualTool # visual module

    print('\ninferencing ...')


    target_frame_path = os.path.join(extract_target_video_path(target_video), 'frames')
    result_save_path = os.path.join(output_path, 'results') 
    os.makedirs(result_save_path, exist_ok=True)

    args = get_experiment_args()

    model = CAMIO.load_from_checkpoint(model_path, args=args) # ckpt
    model = model.cuda()

    db_path = target_frame_path

    # Inference module
    inference = InferenceDB(model, db_path, args.inference_interval) # Inference object
    predict_list, target_img_list, target_frame_idx_list = inference.start() # call start

    # print(predict_list)
    # print(target_img_list)
    # print(target_frame_idx_list)

    # save predict list to csv
    video_name = result_save_path.split('/')[-2]
    
    predict_csv_path = os.path.join(result_save_path, '{}.csv'.format(video_name))
    predict_df = pd.DataFrame({
                    'frame_idx': target_frame_idx_list,
                    'predict': predict_list,
                    'target_img': target_img_list,
                })

    predict_df.to_csv(predict_csv_path)

    return predict_csv_path


def video_editing(predict_csv_path, output_path):
    import os

    # TODO NRS video editing
    # 1. csv_path 입력하여 -> json 생성 (저장 경로 : result_save_path)
    # 2. input 비디오 & results 폴더 복사 (만약 input path, output path 가 동일하면 해당 과정 생략)
    # 3. json 기반 편집 (편집된 결과 저장 경로: )

    print('\nvideo editing ...')
    print(predict_csv_path)
    print(os.path.join(output_path, 'results'))




def video_copy_to_save_dir(target_video, output_path):
    import os
    import shutil

    # inference_result = os.path.join(extract_target_video_path(input_path), 'results/')

    # # move result.csv 
    # print('\nMOVE {} \n==========> {}'.format(inference_result, save_output_path+inference_result))
    # shutil.move(inference_result, save_output_path+inference_result)
    
    # os.makedirs('/'.join(str(output_path+target_video).split('/')[:-1]), exist_ok=True)

    print('\nVideo copying ...')

    # copy target_video
    print('\nCOPY {} \n==========> {}\n'.format(target_video, os.path.join('/'.join(output_path.split('/')[:-1]), target_video.split('/')[-1])))
    shutil.copy2(target_video, os.path.join('/'.join(output_path.split('/')[:-1]), target_video.split('/')[-1]))
    


def check_exist_dupli_video(target_video, output_path):
    import os

    # save_output_path = '/raid/save_output'
    len_save_output_path = len(output_path.split('/'))

    existed_video_list = []
    
    for (root, dirs, files) in os.walk(output_path):
        for file in files:
            # extention 예외 처리 ['mp4', 'mpg']
            if file.split('.')[-1] not in ['mp4', 'mpg']:
                continue

            # '/data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case/L_1/04_GS4_99_L_1_01.mp4'
            existed_video_list.append('/'+'/'.join(os.path.join(root, file).split('/')[len_save_output_path:]))

    for exited_video in existed_video_list:
        if target_video == exited_video:
            return True

    return False
        
    
def main():
    ## 1. input 비디오 목록 읽기
        ## 1-1. 중복 확인 (저장 디렉토리에 결과 존재하는지 확인: 비디오 파일 명 + 알파)
    input_path = '/data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case'
    output_base_path = '/raid/save_output'

    import os
    from core.utils.misc import save_dict_to_csv, prepare_inference_aseets, get_inference_model_path, \
    set_args_per_stage, check_hem_online_mode, clean_paging_chache
    

    for (root, dirs, files) in os.walk(input_path):

        for file in files:
            
            target_video = os.path.join(root, file)
            print('\n', '+++++++++'*10)
            print('*[target video] Processing in {}'.format(target_video))

            # TODO result_save_path 결합 시, 에러 발생 가능성 존재. (output_path : '/raid/' 이런식으로 되어있다면 경로 오류)
            output_path = output_base_path + extract_target_video_path(target_video) # /raid/save_output/data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case/L_1/04_GS4_99_L_1_01
            os.makedirs(output_path, exist_ok=True)
            print('*[output path] {}\n'.format(output_path))
            
            if file.split('.')[-1] not in ['mp4', 'mpg']: # extention 예외 처리 ['mp4', 'mpg']
                continue

            if check_exist_dupli_video(target_video, output_base_path): # True: 중복된 비디오 있음. False : 중복된 비디오 없음.
                print('[DO NOT RUN] ALREADY EXITST IN OUTPUT PATH {}'.format(os.path.join(root, file)))
                continue


            # 1. 비디오 복사 (만약 input path 와 output path 가 동일하면, 해당 과정 생략)
            if input_path != output_base_path:
                video_copy_to_save_dir(target_video, output_path)
            else:
                print("\nINPUT PATH and OUTPUT PATH are same. No duplicating : {}\n".format(input_path))

            ## 2. 비디오 전처리 (frmae 추출) -> 임시 디렉토리
            # frame_cutting(target_video)

            ## 3. inference (비디오 단위) -> 저장 디렉토리 & result csv 생성
            predict_csv_path = inference(target_video, output_path) # model_path 는 args 로 받아도 될 듯.

            ## 4. 비디오 편집 (ffmpep script)
            video_editing(predict_csv_path, output_path)
            
            ## 5. meta_log 파일 생성 & 임시 디렉토리 삭제
            save_meta_log(target_video, output_base_path)

            exit(0)


if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
        sys.path.append(base_path+'/core')
        print(base_path)

    main()