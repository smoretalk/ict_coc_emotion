# ict_coc_emotion
Repository for Whisper-based STT (Speech-to-Text) model and mel-spectrogram classification based emotion recognition

Pre-trained model used for STT: OpenAI Whisper (https://github.com/openai/whisper)

Dataset used to train mel-spectrogram emotion classifier: AI HUB 감성 대화 말뭉치 (https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86)

Uploaded in google drive folder: 
1. AI HUB 감성 대화 말뭉치 음성 데이터 (남:5000, 여:5000, 총 10000 음성 데이터) (wavs.zip) 
2. wav_script (음성별 감성 라벨링: 상처, 기쁨, 불안, 당황, 분노, 슬픔)
3. 감성 대화 말뭉치 데이터를 mel-spectrogram 이미지들로 변환한 데이터 (aihub_mel.zip) 
4. 스펙토그램 변환 데이터 train / val random split 9:1 (image_labels_train.pt, image_labels_val.pt) 
5. 학습된 감성 분류 모델 (resnet_saved_model.pt) 


Training
1. In order to train the speech emotion classifier, make sure to have all paths (data_path, wav_path, directory_path) in the correct location in train.py file
2. run python train.py

How to run Whisper-based STT and emotion classifier based emotion recognition
1. Use the appropriate trained model in load_path 
2. run python run.py
3. record the speech using a microphone, choose a model size (tiny, base, small, medium, large), and hit submit
