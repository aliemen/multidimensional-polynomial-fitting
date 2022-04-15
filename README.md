# Multidimensional Polynomial Fitting
This Python method allows you to fit polynomials of any order in any number of variables to a given data set. Additionally, analogous to Numpy's "polyval", the functionality is given to evaluate the function over a range or to get the callable function.

The simplest use case is for example given by:

```python
fig = plt.figure()
ax = fig.gca(projection='3d')

x, y = np.linspace(-1, 5, 25), np.linspace(-6, 6, 25)
YY, XX = np.meshgrid(x, y, indexing="ij")
ZZ = XX*YY**4 + 2

params, order = poly_n_fit(ZZ, [3, 3], x, y, max_total_order=None)
ZZ_eval, F_fit = poly_n_val(order, params, x, y)

ax.plot_surface(XX, YY, ZZ_eval)
plt.show()
```

If I find the time, I will upload more applications and a sample file in the future.


