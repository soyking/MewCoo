# MewCoo
A hyperspectral remote sensing image classification online tool.

## Dependency

Server:
- [Tornado](https://github.com/tornadoweb/tornado)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [numpy](https://github.com/numpy/numpy)
- [scipy](https://github.com/scipy/scipy)
- [statsmodels](https://github.com/statsmodels/statsmodels)

Webpage:
- [Bootstrap](https://github.com/twbs/bootstrap)
- [Echarts](https://github.com/ecomfe/echarts)
- [JQuery](https://github.com/jquery/jquery)

## How to Run

in main folder :
```python
python multi_classifier.py
```
defalut port : ```8989```

## URL Statement

- <code>/</code>
</br>After inputing image data and selecting classification method (multinomial logistic regression[MLR] or support vector machine[SVM]), you could get classification result reported by text and graphs.And you could select the parameters and the indices of the feature you focus on.

- <code>/stat_feature</code>
</br>Getting original data statistical feature, such as average, variance.

- <code>/rbf_gamma</code>
</br>Trying to find the best gamma of RBF kernel funciton.

- <code>/soft_margin</code>
</br>Trying to find the best soft margin of SVM algorithm.

- <code>/forward_stepwise</code>
</br>Selecting feature indices by forward stepwise.

- <code>/knn</code>
</br>Optimizing classification result by k-nearest neighbors.

- <code>/format</code>
</br>Data format.

## Testing Data

It's in <code>test</code> folder

- all_data for <code>/</code>
- whole_data for <code>/stat_feature</code>
- cv_data for <code>/rbf_gamma</code> and <code>/soft_margin</code>
- fs_data for <code>/forward_stepwise</code>
- result for <code>/knn</code>
