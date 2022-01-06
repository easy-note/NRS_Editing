import os
import pandas as pd


def save_OOB_result_csv(metric, epoch, args, save_path):
    m_keys = list(metric.keys())

    cols = ['Model', 'Epoch', *m_keys]

    save_path = save_path + '/result.csv'
    model_name = args.model
    
    data = [model_name, epoch, *list(metric[key] for key in m_keys)]

    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
        print('Existed file loaded')
        
        new_df = pd.Series(data, index=cols)
        df = df.append(new_df, ignore_index=True)
        print('New line added')
        
    else:
        print('New file generated!')
        df = pd.DataFrame([data],
                    columns=cols
                    ) 

    df.to_csv(save_path, 
            index=False,
            float_format='%.4f')


# inference_flow.py에서 가져옴
def clean_paging_chache():
    import subprocess # for CLEAR PAGING CACHE

    # clear Paging Cache because of I/O CACHE [docker run -it --name cam_io_hyeongyu -v /proc:/writable_proc -v /home/hyeongyuc/code/OOB_Recog:/OOB_RECOG -v /nas/OOB_Project:/data -p 6006:6006  --gpus all --ipc=host oob:1.0]
    print('\n\n\t ====> CLEAN PAGINGCACHE, DENTRIES, INODES "echo 1 > /writable_proc/sys/vm/drop_caches"\n\n')
    subprocess.run('sync', shell=True)
    subprocess.run('echo 1 > /writable_proc/sys/vm/drop_caches', shell=True) ### For use this Command you should make writable proc file when you run docker

def set_args_per_stage(args, ids, stage):
    args.stage = stage

    if ids > 3:
        args.mini_fold = 'general'        
        args.max_epoch = 100
    else:
        args.mini_fold = str(ids)

    return args

def check_hem_online_mode(args):
    if 'online' in args.hem_extract_mode.lower():
        return True
    else:
        return False 

def get_inference_model_path(restore_path):
    # from finetuning model
    import glob
    import os
    
    ckpoint_path = os.path.join(restore_path, '*.ckpt')
    # ckpoint_path = os.path.join(restore_path, 'checkpoints/*.ckpt')
    
    ckpts = glob.glob(ckpoint_path)

    for f_name in ckpts :
        if f_name.find('best') != -1 :
            model_path = f_name
        '''
        if f_name.find('last') != -1 :
            model_path = f_name
        '''
    return model_path

def get_pt_path(restore_path):
    import glob
    import os
    
    model_path = None
    pt_dir = os.path.join(restore_path, '*.pt')
    pts = glob.glob(pt_dir)

    model_path = pts[-1] # load last pt

    return model_path
        

def prepare_inference_aseets(case, anno_ver, inference_fold, save_path):
    from core.utils.prepare import InferenceAssets # inference assets helper (for prepare inference assets)
    from core.utils.prepare import OOBAssets # OOB assets helper (for prepare inference assets)
    from core.utils.parser import FileLoader # file load helper
    
    # OOBAssets
    # assets_sheet_dir = os.path.join(save_path, 'assets')
    # oob_assets = OOBAssets(assets_sheet_dir)
    # oob_assets.save_assets_sheet() # you can save assets sheet
    # video_sheet, annotation_sheet, img_db_sheet = oob_assets.get_assets_sheet() # you can only use assets although not saving
    # video_sheet, annotation_sheet, img_db_sheet = oob_assets.load_assets_sheet(assets_sheet_dir) # you can also load saved aseets

    # InferenceAssets
    inference_assets_save_path = os.path.join(save_path, 'patients_aseets.yaml')
    inference_assets_helper = InferenceAssets(case=case, anno_ver=anno_ver, fold=inference_fold)
    inference_assets = inference_assets_helper.get_inference_assets() # dict (yaml)

    # save InferenceAssets: serialization from python object(dict) to YAML stream and save
    os.makedirs(save_path, exist_ok=True)
    inference_assets_helper.save_dict_to_yaml(inference_assets, inference_assets_save_path)
    
    # load InferenceAssets: load saved inference assets yaml file // you can also load saved patients
    f_loader = FileLoader()
    f_loader.set_file_path(inference_assets_save_path)
    inference_assets = f_loader.load()

    return inference_assets

def save_dict_to_csv(results_dict, save_path):
    import pandas as pd
    from core.utils.parser import FileLoader # file load helper

    results_df = pd.DataFrame.from_dict([results_dict]) # dict to df
    results_df = results_df.reset_index(drop=True)

    merged_df = results_df
    if os.path.isfile(save_path): # append
        f_loader = FileLoader()
        f_loader.set_file_path(save_path)
        saved_df = f_loader.load()

        saved_df.drop(['Unnamed: 0'], axis = 1, inplace = True) # to remove Unmaned : 0 colume

        merged_df = pd.concat([saved_df, results_df], ignore_index=True, sort=False)
        
        merged_df.to_csv(save_path, mode='w')

        print(merged_df)

    merged_df.to_csv(save_path, mode='w')