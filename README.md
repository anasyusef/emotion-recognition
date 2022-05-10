To install the requirements run:

```bash
pip install -r requirements.txt
```

To install the model run:
```bash
cd emotion_recognition
chmod +x get-model.sh
./get-model.sh
```

To install the weights run:

```bash
cd weights
chmod +x get-weights.sh
./get-weights.sh
```

To start emotion recognition on a sample video run:

```bash
python detect_emotions.py --video_src samples/crowd.mp4 --out_video_filename out/crowd.avi
```
