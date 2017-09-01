# Create dataset in the tf.record format
python src/object_detection/create_pascal_tf_record.py \
    --data_dir=data/ \
    --annotations_dir=annotations/ \
    --label_map_path=data/label_map.pbtxt \
    --output_path=data/train.record

# Train a model based on the defined architecture
python src/object_detection/train.py \
    --logstderr \
    --pipeline_config_path=configs/faster_rcnn_resnet101.config \
    --train_dir=data/train_output/rcnn/

# Exporting a trained model as a tf frozen graph
python src/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path configs/faster_rcnn_resnet101.config \
    --checkpoint_path data/train_output/checkpoint.pkl \
    --inference_graph_path src/graphs/faster_rcnn_resnet101_inference_graph.pb

python inference.py
