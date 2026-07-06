project('howdy', 'cpp', license: 'MIT', version: 'beta', meson_version: '>= 0.64.0')

modelsdatadir = get_option('models_data_dir') != '' ? get_option('models_data_dir') : join_paths(get_option('prefix'), get_option('datadir'), 'models-data')
confdir = get_option('config_dir') != '' ? get_option('config_dir') : join_paths(get_option('prefix'), get_option('sysconfdir'), 'howdy')
usermodelsdir = get_option('user_models_dir') != '' ? get_option('user_models_dir') : join_paths(confdir, 'models')
logpath = get_option('log_path')
pythonpath = get_option('python_path')

config_path = join_paths(confdir, 'config.ini')

paths_dict = {
    'config_dir': confdir,
    'models_data_dir': modelsdatadir,
    'user_models_dir': usermodelsdir,
    'log_path': logpath,
    'python_path': pythonpath
}

# We need to keep this order beause howdy-gtk defines the gtk script path which is used later in howdy
subdir('howdy-gtk')
subdir('howdy')