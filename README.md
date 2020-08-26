# trailer_matching
matching the trailer with the movies to get better quality frames and also removes watermark

## Generating a watermark free trailer from the movies itself.
### 


[//]: # (Image References)
[image1]: ./examples/thumbnail.jpg
[image2]: ./examples/thumbnail_video.jpg
[video1]: ./project_video.mp4


### Dataset prepration and method of building the model

This is an unsupervised approach for extracting the frames from the trailer data from the actual movie. This is a very naive and a baseline approach for doing so. The better approach is in building...
Each trailer and it's corresponding movie's videos are being used to extract the frames. Each and every frame is used for processing the model(hence naive), so takes a lot of time and memory for extraction and saving it.
The feature representors are being created from feedforwarding the frames into a CNN based model and those features are being used for matching the trailers and their movies.
As of now, the simple mse loss is calculated to finding the match. 

### Evaluation

As this is an unsupervised approach, it's hard to find the accuracy, precision or recall values, but the coherence in the scenes can be seen in the trailer created. 
Almost every scene in the generated trailer has the same level of coherence(i.e. the frames are from same scene and can totally relate to the global context.) 
Also, there were many frames which were not in the movie itself. For eg:- the starting and the ending part of trailers with the credits scene. 

#### P.S.- The bottom right watarmark is from the software used to merge 2 videos vertically. Above clip - actual trailer, below clip - generated trailer.
#### The real watermark in the trailer is on the above clip's bottom right corner(apple)
[![Watch the video][image2]](https://youtu.be/-BrhNS1Z0PA)


### Room for improvement

We can definately use a new approach and a loss function to find a better match. Also, as seen from the generated trailer, we got the hints that the baseline model can easily detect the scenes and maintains the coherence levels.
So, we can go in direction of extracting the scene changes in the movies and trailers and finding/matching their representations for generating the new trailer can be helpfull.
