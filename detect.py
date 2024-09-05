from ultralytics import YOLO 
import supervision as sv

model = YOLO('yolov8n.pt')
in_filename = 'Traffic.mp4'

video_info = sv.VideoInfo.from_video_path(in_filename)
frames_generator = sv.get_video_frames_generator(in_filename)


# this will handle drawing the bounding boxes and labels
box_annotator = sv.BoxAnnotator()

with sv.VideoSink('/home/grey/Documents/Inter/output.mp4', video_info=video_info) as sink:
    for frame in frames_generator:

        results = model.predict(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # 0.5 stands for threshold i.e any detection that is below this wont be considered valid
        detections = detections[detections.confidence > 0.5]
        
        labels = [f"{model.names[int(class_id)]} {confidence:.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
        
      
        annotated_frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        
      
        sink.write_frame(frame=annotated_frame)