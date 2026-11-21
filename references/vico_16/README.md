# ViCo 16-concept reference images

This first-party folder contains the only external image data required by the
spatial SharedAdaLN experiments. It is intentionally independent of the
DreamMatcher, ARBooth, and other comparison-method submodules.

| Folder | Class prompt | Images |
| --- | --- | ---: |
| `alarm_clock` | `an alarm clock` | 5 |
| `black_cat` | `a grey cat` | 5 |
| `brown_cat` | `a brown cat` | 5 |
| `brown_dog` | `a brown dog` | 5 |
| `brown_dog2` | `a brown dog` | 5 |
| `brown_teddybear` | `a brown teddybear` | 7 |
| `cat_statue` | `a cat statue` | 6 |
| `duck_toy` | `a duck toy` | 4 |
| `elephant_statue` | `an elephant statue` | 5 |
| `fat_dog` | `a fat dog` | 6 |
| `monster_toy` | `a monster toy` | 5 |
| `pink_sunglasses` | `pink sunglasses` | 6 |
| `red_teapot` | `a red teapot` | 5 |
| `red_vase` | `a red vase` | 6 |
| `tortoise_plushy` | `a tortoise plushy` | 12 |
| `wooden_barn` | `a wooden barn` | 7 |

Total: 16 concepts and 94 images.

The fat-dog/grey-cat test reads only `fat_dog/` and `black_cat/`, but all 16
folders are committed so future experiments can run from a normal repository
clone without initializing any comparison-method submodule.
