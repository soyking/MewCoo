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

## URL Statement

- /
</br>After inputing image data and selecting classification method (multinomial logistic regression[MLR] or support vector machine[SVM]), you could get classification result reported by text and graphs.And you could select the parameters and the indices of the feature you focus on.

- /stat_feature
</br>Getting original data statistical feature, such as average, variance.

- /rbf_gamma
</br>Trying to find the best gamma of RBF kernel funciton.

- /soft_margin
</br>Trying to find the best soft margin of SVM algorithm.

- /forward_stepwise
</br>Selecting feature indices by forward stepwise.

- /knn
</br>Optimizing classification result by k-nearest neighbors.

- /format
</br>Data format.

