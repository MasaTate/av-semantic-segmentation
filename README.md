# av-semantic-segmentation

## Folders
config -> configuration files  
dataset -> dataset class files  
metrics -> metrics files  
networks -> model files  
preprocess -> dataset preprocessing files  
utils -> transform functions  

## Usage
既存研究コードの再現設定は，以下のファイルです．  
```
config -> authentic.yaml  
dataset -> sound_cityscapes_auth.py  
training file -> train_audio_seman.py  
```

また，既存研究のコードに，画像変化5%以上のデータのみの条件を追加したのが，以下のファイルです．
```
config -> authentic_thres.yaml  
dataset -> sound_cityscapes_auth_thres.py  
training file -> train_audio_seman.py  
```
