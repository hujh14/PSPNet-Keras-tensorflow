# Evaluate
python src/pspnet.py
-m pspnet50_ade20k \
-l data/places/im_lists/images0.txt \
-d data/places/images \
-o data/places/pspnet_predictions/images0 \
--id 0