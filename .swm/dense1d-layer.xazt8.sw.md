---
id: xazt8
title: Dense1d Layer
file_version: 1.1.1
app_version: 1.1.0
---

Dense 1d layer that every NN has.

# Creation functions

<br/>

From is used when you need to control the creation of every part of the layer.
<!-- NOTE-swimm-snippet: the lines below link your snippet to Swimm -->
### 📄 src/layers.rs
```renderscript
14         pub fn from(
15             activation: fn(Array1<f64>) -> Array1<f64>,
16             weights: Array2<f64>,
17             bias: Array1<f64>,
18         ) -> Self {
```

<br/>

<br/>

<br/>

This file was generated by Swimm. [Click here to view it in the app](https://app.swimm.io/repos/Z2l0aHViJTNBJTNBZHVja3ktbGVhcm4lM0ElM0FKYWNrMTc0MzI=/docs/xazt8).