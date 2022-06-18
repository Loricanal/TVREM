import wave, math, contextlib
import speech_recognition as sr
from moviepy.editor import AudioFileClip
import os
import sys
import json
import moviepy.editor as mp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import moviepy
from datetime import datetime
import shutil
import subprocess
import shlex



# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

try:    
    input_folder = sys.argv[1]
except:
    raise("You have to specify the input folder")
try:
    output_folder = sys.argv[2]
except:
    raise("You have to specify the output folder")


output_files = [file for file in os.listdir(output_folder)]

print("Output files len:",len(output_files))

todo_files = [file for file in os.listdir(input_folder) if "._" != file[:2] and file.endswith(".mp4") and file.replace('.mp4','.txt') not in output_files]

print("Todo files len:",len(todo_files))
credentials_apis = json.load(open('credentials.json'))
credentials_string_google = json.dumps(credentials_apis['google'][0])



def getTranscriptFromGoogleApi(video_file_name,output_path):
    try:
        os.remove('TEMP.mp4')
    except:
        pass
    fmt = '%Y-%m-%d %H:%M:%S'
    temp_audio = "temp_audio/"
    try:
        os.makedirs(temp_audio)
    except:
        pass
    try:
        os.makedirs("tmp_chunks/")
    except:
        pass
    # Create an object by passing the location as a string
    video = moviepy.editor.VideoFileClip(video_file_name)
    # Contains the duration of the video in terms of seconds
    num_seconds_video = int(video.duration)
    print("The video is {} seconds".format(num_seconds_video))
    l=list(range(0,num_seconds_video+1,60))
    diz={}
    print("Total_number of chunks:",len(l)-1)

    K = int(len(l)/8)

    tstamp1 = datetime.now()
    try:
        ffmpeg_extract_subclip(video_file_name, l[0]-2*(l[0]!=0), l[0+1], targetname="tmp_chunks/cut{}.mp4".format(0+1))
    except:
        print("CONVERSION",video_file_name)
        cmd = "ffmpeg -i FILE -vcodec copy -acodec aac TEMP.mp4".replace("FILE",video_file_name)
        try:
            p = subprocess.Popen(cmd, shell=True, stdout=None)
            p.wait()
            print("retcode =", p.returncode)
            #nput(61)
            flag = False
        except:
            flag = True
            print("Flag",video_file_name)
            #input(62)
        if p.returncode != 0:
            #print(chr(27) + "[2J")
            return 0
        #p = subprocess.Popen(cmd, shell=True, stdout=True)
        #p.wait()
        video_file_name = "TEMP.mp4"
        ffmpeg_extract_subclip(video_file_name, l[0]-2*(l[0]!=0), l[0+1], targetname="tmp_chunks/cut{}.mp4".format(0+1))
    keys = list()
    count_err = 0
    for i in range(len(l)-1):
        #enablePrint()
        print("Chunk:",i)
        #blockPrint()
        try:
            ffmpeg_extract_subclip(video_file_name, l[i]-2*(l[i]!=0), l[i+1], targetname="tmp_chunks/cut{}.mp4".format(i+1))
            clip = mp.VideoFileClip(r"tmp_chunks/cut{}.mp4".format(i+1)) 
            clip.audio.write_audiofile(temp_audio+r"converted{}.wav".format(i+1))
            r = sr.Recognizer()
            audio = sr.AudioFile(temp_audio+r"converted{}.wav".format(i+1))
            with audio as source:
                r.adjust_for_ambient_noise(source)  
                audio_file = r.record(source)
                #enablePrint()
                print("Recognize")
                #blockPrint()
                result = r.recognize_google(audio_file)
                #enablePrint()
                print("Recognized")
                #blockPrint()
                k = i+1
                keys.append(k)
                #print(5)
                diz['chunk{}'.format(k)]=result
                #print(6)
                os.remove(r"tmp_chunks/cut{}.mp4".format(i+1))
                #print(7)
                os.remove(temp_audio+r"converted{}.wav".format(i+1))
        except:
            if count_err < K:
                print("AIAIAI")
                count_err += 1
            else:
                print("ERROR")
                print("Count error:",count_err)
                #enablePrint()
                raise
    try:
        os.remove('TEMP.mp4')
    except:
        pass
        #enablePrint()
    #print("ALWEE")
    shutil.rmtree("tmp_chunks/")
    shutil.rmtree(temp_audio)
    #os.rmdir("tmp_chunks/")
    #os.rmdir(temp_audio)
    l_chunks=[diz['chunk{}'.format(i)] for i in keys]
    text=('\n'.join(l_chunks)).replace("Recognized Speech:","").replace("\n","")
    print("Fuck")
    with open(output_path,mode ='w') as file:
        file.write("Recognized Speech:")
        file.write("\n")
        file.write(text)
        tstamp2 = datetime.now()
        if tstamp1 > tstamp2:
            td = tstamp1 - tstamp2
        else:
            td = tstamp2 - tstamp1
        td_mins = int(round(td.total_seconds() / 60))
        print("Finally ready in minutes:",td_mins)



max_failures = 1

for file in todo_files:
    print("Computing transcript for:",file)
    video_file_name = os.path.join(input_folder,file)
    output_path = os.path.join(output_folder,file.replace(".mp4",".txt"))
    count_failures = 0
    while count_failures < max_failures:
        try:
            getTranscriptFromGoogleApi(video_file_name,output_path)
            count_failures = max_failures
        except Exception as e:
            print("Exception:",str(e))
            count_failures += 1
            print("Count failures:",count_failures)
    print("DONE:",file)
    output_files = [f for f in os.listdir(output_folder)]
    print("Output files len:",len(output_files))






