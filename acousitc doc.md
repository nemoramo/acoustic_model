#acousitc doc

##model

**Structure:**

![Screenshot from 2019-07-17 17-34-56](/home/jilei/Pictures/acousitc/Screenshot from 2019-07-17 17-34-56.png)



```python
"""
model input shape:
	[batch_size, channel, audio_length, feature_length]
"""

print(model(torch.randn(3, 1, 1600, 200)).shape)

"""
out:
	shape: torch.Size([200, 3, 1000])
"""
```



