
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



# 22.01.07 hg modify, 기존 target_video 내부에 생성 => 설정된 frame_save_path 생성되도록 변경
def frame_cutting(target_video, frame_save_path):
    import sys
    import os    

    print('\nframe cutting ... ')

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    
    sys.path.append(base_path)
    sys.path.append(base_path+'core')

    from core.utils.ffmpegHelper import ffmpegHelper

    # save_path 
    os.makedirs(frame_save_path, exist_ok=True)

    # frame cutting -> save to '$ frame_save_path~/frame-000000000.jpg'
    ffmpeg_helper = ffmpegHelper(target_video, frame_save_path)
    ffmpeg_helper.cut_frame_total()

    return frame_save_path


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


# 22.01.17 hg modify, InfernceDB독립사용을 위해 target_video 변수를 target_dir로 변경하고 infernce_interval parameter를 추가하였습니다.
def inference(target_dir, inference_interval, result_save_path, model_path):
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
    # 22.01.17 hg modify, 해당 inference 함수 사용시 사용하지 않는 module import문을 제거하였습니다.

    print('\ninferencing ...')

    os.makedirs(result_save_path, exist_ok=True)

    # 22.01.17 hg comment, args는 model 불러올떄만 사용하고, InferencDB사용시에 args 사용을 제거하였습니다.
    args = get_experiment_args()

    model = CAMIO.load_from_checkpoint(model_path, args=args) # ckpt
    model = model.cuda()

    # Inference module
    inference = InferenceDB(model, target_dir, inference_interval) # 22.01.17 hg modify, InferenceDB 사용시 기존 args.inference_interval 에서 function param으로 변경하였습니다.
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


def get_event_sequence_from_csv(predict_csv_path):
    import pandas as pd

    return pd.read_csv(predict_csv_path)['predict'].values.tolist()


def get_video_meta_info_from_ffmpeg(video_path):
    from core.utils.ffmpegHelper import ffmpegHelper

    print('\n\n \t\t <<<<< GET META INFO FROM FFMPEG >>>>> \t\t \n\n')

    ffmpeg_helper = ffmpegHelper(video_path)
    fps = ffmpeg_helper.get_video_fps()
    video_len = ffmpeg_helper.get_video_length()
    width, height = ffmpeg_helper.get_video_resolution()

    return fps, video_len, width, height

def extract_video_clip(video_path, results_dir, start_time, duration):
    from core.utils.ffmpegHelper import ffmpegHelper

    ffmpeg_helper = ffmpegHelper(video_path, results_dir)
    ffmpeg_helper.extract_video_clip(start_time, duration)


def parse_clips_paths(clips_root_dir):
    import os
    import glob
    from natsort import natsort

    target_clip_paths = glob.glob(os.path.join(clips_root_dir, 'clip-*.mp4'))
    target_clip_paths = natsort.natsorted(target_clip_paths)

    save_path = os.path.join(clips_root_dir, 'clips.txt')

    logging = ''

    for clip_path in target_clip_paths:
        txt = 'file \'{}\''.format(os.path.abspath(clip_path))
        logging += txt + '\n'
    
    print(logging)

    # save txt
    with open(save_path, 'w') as f :
        f.write(logging)

    return save_path


def clips_to_video(clips_root_dir, merge_path):
    from core.utils.ffmpegHelper import ffmpegHelper

    # parsing clip video list 
    input_txt_path = parse_clips_paths(clips_root_dir)

    ffmpeg_helper = ffmpegHelper("dummy", "dummy")
    ffmpeg_helper.merge_video_clip(input_txt_path, merge_path)


## 4. 22.01.07 hg new add, save annotation by inference (hvat form)
def report_annotation(frameRate, totalFrame, width, height, name, event_sequence, inference_interval, result_save_path):
    from core.utils.report import ReportAnnotation
    from core.utils.misc import get_nrs_frame_chunk, get_current_time
    
    ### static meta data ###
    createdAt = get_current_time()[0]
    updatedAt = createdAt
    _id = "61efa"
    annotationType = "NRS"
    annotator = "30"
    label = {"1": "NonRelevantSurgery"}
    ### ### ### ### ### ###

    nrs_frame_chunk = get_nrs_frame_chunk(event_sequence, inference_interval)

    annotation_report = ReportAnnotation(result_save_path) # load Report module

    # set meta info
    annotation_report.set_total_report(totalFrame, frameRate, width, height, _id, annotationType, createdAt, updatedAt, annotator, name, label)
    
    # add nrs annotation info
    nrs_cnt = len(nrs_frame_chunk)
    for i, (start_frame, end_frame) in enumerate(nrs_frame_chunk, 1):
        # print('\n\n[{}] \t {} - {}'.format(i, start_frame, end_frame))

        # check over totalFrame on last annotation (because of quntization? when set up inference_interval > 1)
        if nrs_cnt == i and end_frame >= totalFrame: 
            end_frame = totalFrame - 1

        annotation_report.add_annotation_report(start_frame, end_frame, code=1)

    annotation_report.save_report()

## 5. 22.01.07 hg write, inference_interval, video_fps는 clips의 extration time 계산시 필요
def video_editing(video_path, event_sequence, editted_video_path, inference_interval, video_fps):
    import os
    from core.utils.misc import get_rs_time_chunk

    print('\nvideo editing ...')

    # temp process path
    temp_process_dir = os.path.dirname(editted_video_path) # clips, 편한곳에 생성~(현재는 edit되는 비디오 밑에 ./clips에서 작업)
    temp_clip_dir = os.path.join(temp_process_dir, 'clips')
    os.makedirs(temp_clip_dir, exist_ok=True)

    # 0. inference vector(sequence) to rs chunk
    target_clipping_time = get_rs_time_chunk(event_sequence, video_fps, inference_interval)


    print('\n\n \t\t <<<<< EXTRACTING CLIPS >>>>> \t\t \n\n')
    # 1. clip video from rs chunk
    for i, (start_time, duration) in enumerate(target_clipping_time, 1):
        print('\n\n[{}] \t {} - {}'.format(i, start_time, duration))
        extract_video_clip(video_path, temp_clip_dir, start_time, duration)

    # 2. merge video
    print('\n\n \t\t <<<<< MERGEING CLIPS >>>>> \t\t \n\n')
    clips_to_video(clips_root_dir = temp_clip_dir, merge_path = editted_video_path)

    # 3. TODO: delete temp process path (delete clips)


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

# predict csv to applied-pp predict csv
def apply_post_processing(predict_csv_path, seq_fps):
    import os
    import pandas as pd
    from core.api.post_process import FilterBank
    
    # 1. load predict df
    predict_df = pd.read_csv(predict_csv_path, index_col=0)
    event_sequence = predict_df['predict'].tolist()

    # 2. apply pp sequence
    #### use case 1. best filter
    fb = FilterBank(event_sequence, seq_fps)
    best_pp_seq_list = fb.apply_best_filter()

    #### use case 2. custimize filter
    '''
    best_pp_seq_list = fb.apply_filter(event_sequence, "opening", kernel_size=3)
    best_pp_seq_list = fb.apply_filter(best_pp_seq_list, "closing", kernel_size=3)
    '''

    predict_df['predict'] = best_pp_seq_list

    # 3. save pp df
    d_, f_ = os.path.split(predict_csv_path)
    f_name, _ = os.path.splitext(f_)

    results_path = os.path.join(d_, '{}-pp.csv'.format(f_name)) # renaming file of pp
    predict_df.to_csv(results_path)

    return results_path
    
def main():
    ## 1. input 비디오 목록 읽기
        ## 1-1. 중복 확인 (저장 디렉토리에 결과 존재하는지 확인: 비디오 파일 명 + 알파)
    # input_path = '/data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case'
    # output_base_path = '/raid/save_output'
    input_path = '/data3/DATA/IMPORT/211220/12_14/TEST_24case'
    output_base_path = '/data3/Public/save_output2'

     # 22.01.07 hg comment, 우선 모델 불어올때만 args가 이용되고, 나머지는 최대한 args에 의존되지 않도록 처리하기 위해 변수화 시켰습니다.
     # 추후 args로 받아올 경우 해당 변수를 args. 로 초기화
    seq_fps = 1 # pp module (1 fps 로 inference) - pp에서 사용 (variable이 고정되어 30 fps인 비디오만 사용하는 시나이오로 적용, 비디오에 따라 유동적으로 var이 변하도록 계산하는게 필요해보임)
    inference_interval = 30 # frame inference interval - 전체사용 (variable이 고정되어 30 fps인 비디오를 1fps로 Infeence 하는 시나리오로 적용, 비디오에 따라 유동적으로 var이 변하도록 계산하는게 필요해보임)
    model_path = '/NRS_EDITING/logs_sota/mobilenetv3_large_100-theator_stage100-offline-sota/version_4/checkpoints/epoch=60-Mean_metric=0.9875-best.ckpt'


    import os
    import glob

    for (root, dirs, files) in os.walk(input_path):

        for file in files:
            
            target_video = os.path.join(root, file)
            print('\n', '+++++++++'*10)
            print('*[target video] Processing in {}'.format(target_video))

            # TODO result_save_path 결합 시, 에러 발생 가능성 존재. (output_path : '/raid/' 이런식으로 되어있다면 경로 오류)
            output_path = output_base_path + extract_target_video_path(target_video) # /raid/save_output/data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case/L_1/04_GS4_99_L_1_01
            os.makedirs(output_path, exist_ok=True)
            print('*[output path] {}\n'.format(output_path))
            
            if file.split('.')[-1].lower() not in ['mp4', 'mpg']: # extention 예외 처리 ['mp4'] # 22.01.07 hg comment, sanity한 ffmpeg module 사용을 위해 우선 .mp4 만 사용하는 것이 좋을 것 같습니다.
                continue

            if check_exist_dupli_video(target_video, output_base_path): # True: 중복된 비디오 있음. False : 중복된 비디오 없음.
                print('[DO NOT RUN] ALREADY EXITST IN OUTPUT PATH {}'.format(os.path.join(root, file)))
                continue
            
            # 0. get video meta info
            frameRate, totalFrame, width, height = get_video_meta_info_from_ffmpeg(target_video) # from ffmpeg
            video_name = os.path.splitext(os.path.basename(target_video))[0]

            # 1. 비디오 복사 (만약 input path 와 output path 가 동일하면, 해당 과정 생략)
            if input_path != output_base_path:
                video_copy_to_save_dir(target_video, output_path)
            else:
                print("\nINPUT PATH and OUTPUT PATH are same. No duplicating : {}\n".format(input_path))

            # 22.01.07 hg comment, 각 사용함수 내부에 결과가 저장되는 dir이 따로따로 조합되어 결과 저장경로를 하위까지 조합하여 parameter로 넘겨주었습니다. (~/frames, ~/results)
            
            ## 2. 비디오 전처리 (frmae 추출) -> 임시 디렉토리
            frame_save_path = frame_cutting(target_video, frame_save_path = os.path.join(output_path, 'frames')) # 22.01.07 hg modify, output_path에 처리되도록 변경 및 processed dir 반환
            frame_save_path = os.path.join(output_path, 'frames')

            ## 3. inference (비디오 단위) -> 저장 디렉토리 & result csv 생성 
            # 22.01.07 hg modify, target_video -> frame_save_path로 변경, inference_interval 추가
            predict_csv_path = inference(target_dir = frame_save_path, inference_interval = inference_interval, result_save_path = os.path.join(output_path, 'results'), model_path = model_path) # model_path 는 args 로 받아도 될 듯.

            # prepare 1. csv to sequence vector
            event_sequence = get_event_sequence_from_csv(predict_csv_path)

            # prepare 2. apply PP module
            predict_csv_path = apply_post_processing(predict_csv_path, seq_fps) # csv vector to applid pp csv vector
            event_sequence = get_event_sequence_from_csv(predict_csv_path) # reload - applied pp csv vector

            # prepare 3. check unmatching total frame
            frame_cnt = len(glob.glob(os.path.join(frame_save_path, '*.jpg')))
            if frame_cnt < totalFrame:
                print('>>>>> UNMATCH FRAME CNT <<<<< \t extrated frame_cnt : {} \t < \t totalFrame by ffmpeg : {} '.format(frame_cnt, totalFrame))  # TODO - logging
                totalFrame = frame_cnt           

            ## 4. 22.01.07 hg new add, save annotation by inference (hvat form)
            report_annotation(frameRate, totalFrame, width, height, video_name, event_sequence, inference_interval, os.path.join(output_path, 'results', '{}-annotation_by_inference.json'.format(video_name)))

            ## 5. 비디오 편집 (ffmpep script)
            video_editing(target_video, event_sequence, os.path.join(output_path, 'results', '{}-edit.mp4'.format(video_name)), inference_interval, frameRate)
            
            ## 6. meta_log 파일 생성 & 임시 디렉토리 삭제
            save_meta_log(target_video, output_base_path)



if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
        sys.path.append(base_path+'/core')
        print(base_path)

    main()