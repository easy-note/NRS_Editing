
# input video path (임시 테스트용) : /NAS/nas19/DATA/IMPORT/211220/12_14/gangbuksamsung_127case/L_1
# output dir path (임시 테스트용) : /raid/NRS_Editing

def get_input_video(input_path, output_path):
    import os

    for (root, dirs, files) in os.walk(input_path):
        for file in files:
            print(file)




def main():
    import os
    import glob

    ## 1. input 비디오 목록 읽기
        ## 1-1. 중복 확인 (저장 디렉토리에 결과 존재하는지 확인: 비디오 파일 명 + 알파)
    input_video_path = '/data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case/L_1'
    output_dir_path = '/raid'

    get_input_video(input_video_path, output_dir_path)

    ## 2. 비디오 전처리 (frmae 추출) -> 임시 디렉토리
    ## 3. inference (비디오 단위) -> 저장 디렉토리 & result json 생성
    ## 4. 비디오 편집 (ffmpep script)
    ## 5. 임시 디렉토리 삭제
        ## 5-1. log 파일 생성


if __name__ == '__main__':
    main()