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



##Note

1. **Iter 1**

   **BaseModel:** Deep Conv Net

   * Modify: 接2层Bi-LSTM -> not work
   * Modify: 接4层GRU -> not work
   * Modify: 对RNN使用LayerNorm -> not work

   **Main Problem:** Train Loss 下降到一定程度后，Evaluate Loss不降反升(降到10以下开始飙升到30 )

   **Possible Reason:** overfit

   * Modify: 将dropout由0.1提升至0.5 -> not work

   **Question:** 	

   ​	Shallow Convolution + large kernel(e.g. 40×20) + large(e.g. 5) stride

   ​     = Deep Convolution + small kernel(e.g. 3×3 or 5×5) + 1 stride ?

   **Solution:** 

   ​	dropout改为dropout2d提升显著。loss缓慢下降，虽然波动幅度客观，但是学习效果符合优化特征。dropout2d: 对C dimension 做dropout(一次性抛弃整个channel)

2. **Iter 2**

   **BaseModel:** Deep Conv Net + Multilayer RNN

   * Modify: dropout2d 放在RNN batchnorm 后面 ？
   * 在最后一层使用Attention机制 ？
   * LAS model

   ​

