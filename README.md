### Usage

`python sam.py --block_layers=3 --blocks=2`

### About
 
- This script tells us what kind of changes we are seeing in the gradients when we introduce skip connections to our model
- For the model I'm simply using a stack of FeedForward layers with ReLU activation function

### Output Sample:
- When we simply run `python sam.py`:
  
  ```
  Following are the changes in grads when we make use of skip connections: 
     layers.0.0.linear_layer.weight : 0.00320375 => 0.05742524
     layers.1.0.linear_layer.weight : 0.00889227 => 0.17762329
     layers.2.0.linear_layer.weight : 0.01780202 => 0.20498182
     layers.3.0.linear_layer.weight : 0.01799066 => 0.25682366
     layers.4.0.linear_layer.weight : 0.03587941 => 0.38802755
     final_layer.weight : 0.53811127 => 1.39856958
  ```
