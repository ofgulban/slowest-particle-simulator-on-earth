<img src="/visuals/animation_01_opt.gif" width=200 align="right" />

# Slowest particle simulator on earth
Just a fun project to learn about particle simulations. Developed for applying particle physics to magnetic resonance images (nifti file format).

## Dependencies
**[Python 3](https://www.python.org/)**

| Package                                                    | Tested version |
|------------------------------------------------------------|----------------|
| [NumPy](http://www.numpy.org/)                             | 1.17.2         |
| [Matplotlib](https://matplotlib.org/)                      | 3.1.1          |
| [Nibabel](https://nipy.org/nibabel/)                       | 2.2.1          |

## Installation
```
pip install slowest_particle_simulator_on_earth
```

## Usage
Type the following command on your command line:
```
slowest_particle_simulator_on_earth /path/to/image.nii.gz --slice_number 165 --thr_min 200 --thr_max 500
```

*Note:* You can select different slice numbers. For now the slices can only be chosen on one axis. I am going to make this more flexible later.

## Making a gif
*slowest_particle_simulator_on_earth* creates individual pictures which can be compiled into an animated gif. I use the following command (on linux) to convert the frames into animated gifs:
```
ffmpeg -r 1 -i /path/to/export/frame_%03d.png -pix_fmt yuv420p -r 30 out.mp4
```

## Support
Please use [GitHub issues](https://github.com/ofgulban/slowest-particle-simulator-on-earth/issues) for questions, bug reports or feature requests.

## License
This project is licensed under [BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause).
