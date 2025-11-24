from open_cctv import capture_and_save_photo

# replace with CCTV camera's URL e.g. url = 'rtsp://192.168.1.41:554/mcast/11'
url = 'rtsp://admin:csimAIT5706@192.168.6.100:554/Streaming/Channels/101/' 

capture_and_save_photo(url=url, save_dir='cam2', filename_prefix='cam2', ext='png', delay=1, window_name='Camera 1', image_number=13)