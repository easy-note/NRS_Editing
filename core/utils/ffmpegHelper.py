from subprocess import Popen, PIPE

class ffmpegHelper():
    def __init__(self, video_path, results_dir='dummy'):
        self.EXCEPTION_NUM = -100
        self.video_path = video_path
        self.results_dir = results_dir # only use in [_process_cmd]

        
    def _attr_cmd(self, attr):
        
        attr_cmd = {
            'fps': ['ffmpeg', '-i', self.video_path, '2>&1', '|', 'sed', '-n', '"s/.*, \(.*\) fp.*/'+'\\'+'1/p"'],
            'video_length': ['ffprobe', '-v', 'error', '-select_streams v:0', '-count_packets', '-show_entries stream=nb_read_packets', '-of', 'csv=p=0', self.video_path]
        }

        cmd_list = attr_cmd[attr]

        return ' '.join(cmd_list)
    
    def _process_cmd(self, process, **args):
        process_cmd = {
            'cut_frame_total': ['ffmpeg', '-i', self.video_path, '-start_number', '0', '-vsync', '0', '-vf', 'scale=512:512', self.results_dir + '/frame-%010d.jpg'], ## vsync 0 으로 두었을 때, 끝까지 안 잘리는 경우가 있었는데, 고민이네요..
            'cut_frame_1fps': ['ffmpeg', '-i', self.video_path, '-s', '224x224', '-vf', 'fps=1', self.results_dir + '/frame-%010d.jpg'],
            'extract_frame_by_index': ['ffmpeg', '-i', self.video_path, '-vf', '"select=eq(n\,{})"'.format(args.get('frame_index', -100)), '-vframes', '1', self.results_dir + '/extracted_frame-{}.jpg'.format(args.get('frame_index', -100))], # https://superuser.com/questions/1009969/how-to-extract-a-frame-out-of-a-video-using-ffmpeg
            'extract_frame_by_time': ['ffmpeg', '-i', self.video_path, '-ss', '{}'.format(args.get('time', -100)), '-frames:v', '1', self.results_dir + '/extracted_frame-{}.jpg'.format(args.get('time', -100))],
        }

        cmd_list = process_cmd[process]

        return ' '.join(cmd_list)
    
    def _cmd_call(self, cmd):
        try :
            procs_list = [Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)]

            for proc in procs_list:
                print('ing ...')
                # proc.wait() # communicate() # 자식 프로세스들이 I/O를 마치고 종료하기를 기다림
                out = proc.communicate()
                print("Processes are done")

            out = out[0].decode('UTF-8').rstrip() # byte to str
        
        except:
            out = self.EXCEPTION_NUM

        return out
        
    #### user func ####
    def get_video_fps(self):
        cmd = self._attr_cmd('fps')

        print('GET VIDEO FPS USIGN FFMPEG : {}'.format(cmd), end= ' ')

        fps = self._cmd_call(cmd)
            
        fps = float(fps)
        return fps

    def get_video_length(self):
        
        cmd = self._attr_cmd('video_length')

        print('GET VIDEO LENGTH USIGN FFPROBE : {}'.format(cmd), end= ' ')

        video_len = self._cmd_call(cmd)

        video_len = int(video_len)
        return video_len

    def cut_frame_1fps(self):
        cmd = self._process_cmd('cut_frame_1fps')
        
        print('CUT FRAME 1FPS : {}'.format(cmd), end= ' ')

        self._cmd_call(cmd)

    def cut_frame_total(self):
        cmd = self._process_cmd('cut_frame_total')

        print('CUT FRAME TOTAL : {}'.format(cmd), end= ' ')

        self._cmd_call(cmd)

    def extract_frame_by_index(self, frame_index):
        # frame_index= 500
        cmd = self._process_cmd('extract_frame_by_index', frame_index=frame_index)

        print(cmd)
        exit(0)
        
        print('EXTRACT FRAME BY INDEX : {}'.format(cmd), end= ' ')

        self._cmd_call(cmd)

    def extract_frame_by_time(self, time):
        ### frame_time= 00:00:05.01
        cmd = self._process_cmd('extract_frame_by_time', time=time)
        
        print('EXTRACT FRAME BY TIME : {}'.format(cmd), end= ' ')

        self._cmd_call(cmd)