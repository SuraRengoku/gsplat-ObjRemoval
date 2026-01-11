https://github.com/jonstephens85/gsplat_3dgut

install Imagemagick colmap
create a conda environment for gsplat


int the data folder, create a folder of your dataset with anyname xxx, then put images into xxx/images

if you want to downsize the images, run:
mkdir images_2
magick mogrify -path images_2 -resize 50%% images\.*jpg(png)

then we have to exploit the geometry information by using colmap:
mkdir sparse

colmap feature_extractor --database_path database.db --image_path images --ImageReader.camera_model SIMPLE_PINHOLE --ImageReader.single_camera 1

colmap exhaustive_matcher --database_path database.db

colmap mapper --database_path database.db --image_path images --output_path sparse

all information collected, ready to train:
python examples/simple_trainer.py default --data_dir data/xxx/ --data_factor x --result_dir results/xxx

if you want to generate depth map after training:
python examples/simple_trainer.py default --data_dir data/xxx/ --data_factor x --result_dir results/xxx --ckpt results/xxx/ckpts/ckpt_30000_rank0.pt --disable_viewer

if you want to generate depth map during training:
python examples/simple_trainer.py default --data_dir data/xxx/ --data_factor x --result_dir results/xxx --save_train_depths --save_ply