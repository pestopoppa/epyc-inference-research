# GLM-5.2 C-CRAB Accept-Control Review Packet

This packet is a review aid only. It does not modify `signoff` fields,
does not claim operator approval, and does not make the accept controls decision-grade.

- Rows: `24`
- Source schema: `glm52_ccrab_accept_control_audit_packet.v1`
- Excerpt chars per task/candidate: `1800`

| # | row_id | repo | PR | machine recommendation | concerns |
|---:|---|---|---:|---|---|
| 1 | `nearmiss-v1:c-crab:00710c9c18cd10fb` | scikit-learn/scikit-learn | 29401 | hard_accept_candidate | none |
| 2 | `nearmiss-v1:c-crab:0110087826d99378` | sympy/sympy | 18410 | hard_accept_candidate | none |
| 3 | `nearmiss-v1:c-crab:04b3b67ccf17ffe1` | python/mypy | 3005 | hard_accept_candidate | Patch includes a submodule commit update; reviewer should ensure the packet preserves the intended submodule transition. |
| 4 | `nearmiss-v1:c-crab:04d390e945dd768e` | scikit-learn/scikit-learn | 10099 | hard_accept_candidate | none |
| 5 | `nearmiss-v1:c-crab:060c7e12cfd0cb06` | dask/dask | 10723 | hard_accept_candidate | none |
| 6 | `nearmiss-v1:c-crab:0735945503ef9330` | ManimCommunity/manim | 509 | hard_accept_candidate | none |
| 7 | `nearmiss-v1:c-crab:08cafcb6483d8389` | pandas-dev/pandas | 33962 | hard_accept_candidate | candidate_redacted_long_digit_runs is true; visible redaction appears compatible with diff index metadata rather than substantive code. |
| 8 | `nearmiss-v1:c-crab:09584d0209952576` | numpy/numpy | 14924 | hard_accept_candidate | candidate_redacted_long_digit_runs is true; visible redaction appears limited to diff index metadata. |
| 9 | `nearmiss-v1:c-crab:0b5adcf2e8a30f49` | scikit-learn/scikit-learn | 23548 | hard_accept_candidate | none |
| 10 | `nearmiss-v1:c-crab:0c6318021a8a500b` | pandas-dev/pandas | 24968 | hard_accept_candidate | none |
| 11 | `nearmiss-v1:c-crab:0e31e881b1af8ab5` | reflex-dev/reflex | 4406 | hard_accept_candidate | none |
| 12 | `nearmiss-v1:c-crab:0f20280ccef865cb` | home-assistant/core | 142977 | hard_accept_candidate | none |
| 13 | `nearmiss-v1:c-crab:10070430d41b73e9` | sympy/sympy | 24666 | hard_accept_candidate | candidate_redacted_long_digit_runs is true; reviewer should confirm redaction does not hide a significant literal in the expected Piecewise expression. |
| 14 | `nearmiss-v1:c-crab:12dedf2f36029e2c` | scikit-learn/scikit-learn | 29442 | hard_accept_candidate | none |
| 15 | `nearmiss-v1:c-crab:1600ca8239e2f6e0` | python/mypy | 11143 | hard_accept_candidate | candidate_redacted_long_digit_runs is true; redaction may affect path or fixture literal review and should be checked manually before authoritative signoff. |
| 16 | `nearmiss-v1:c-crab:19fd2cb501691488` | pandas-dev/pandas | 61286 | hard_accept_candidate | none |
| 17 | `nearmiss-v1:c-crab:1a3868334fafed91` | pandas-dev/pandas | 18831 | hard_accept_candidate | none |
| 18 | `nearmiss-v1:c-crab:1a64c956e9fceeda` | pandas-dev/pandas | 18604 | hard_accept_candidate | none |
| 19 | `nearmiss-v1:c-crab:1af8c54719aff460` | spyder-ide/spyder | 11708 | hard_accept_candidate | none |
| 20 | `nearmiss-v1:c-crab:200003ca11cb7699` | jina-ai/serve | 2037 | hard_accept_candidate | candidate_redacted_long_digit_runs is true; visible redaction appears limited to generated or diff metadata, but the Dockerfile fixture should be manually reviewed before final signoff. |
| 21 | `nearmiss-v1:c-crab:20e3c97d771762dd` | numba/numba | 4282 | hard_accept_candidate | none |
| 22 | `nearmiss-v1:c-crab:24fbdf7de1c6fa44` | scipy/scipy | 10410 | hard_accept_candidate | none |
| 23 | `nearmiss-v1:c-crab:28cdbe123c730a18` | pandas-dev/pandas | 28802 | hard_accept_candidate | none |
| 24 | `nearmiss-v1:c-crab:29de0af708959323` | dbt-labs/dbt-core | 1453 | hard_accept_candidate | none |

## 1. `nearmiss-v1:c-crab:00710c9c18cd10fb`

- Instance: `scikit-learn__scikit-learn-29401@819e167`
- Repo / PR: `scikit-learn/scikit-learn` / `29401`
- Candidate chars: `2968`
- Candidate redacted long digit runs: `False`
- Machine recommendation: `hard_accept_candidate`
- Machine reason: Candidate directly neutralizes global dataframe set_output warnings for TransformedTargetRegressor by forcing the internally-created target FunctionTransformer back to default output, with pandas and polars regression tests plus release note.
- Format concerns: none

**Task Excerpt**

````
FIX `TransformedTargetRegressor` warns when `set_output` expects dataframe

TransformedTargetRegressor warns about set_output set to pandas
### Describe the bug

If `set_output` is set to `"pandas"`, `TransformedTargetRegressor` warns unnecessarily.

### Steps/Code to Reproduce

```python
import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

set_config(transform_output="pandas")
X, y = make_regression()
y = np.abs(y) + 1
TransformedTargetRegressor(
    regressor=LinearRegression(),
    func=np.log,
    inverse_func=np.exp,
).fit(X, y)
```

### Expected Results

No warning.

### Actual Results

3 times the same warning:
```
python3.11/site-packages/sklearn/preprocessing/_function_transformer.py:303: UserWarning: When `set_output` is configured to be 'pandas', `func` should return a pandas DataFrame to follow the `set_output` API  or `feature_names_out` should be defined.
  warnings.warn(warn_msg.format("pandas"))
```

### Versions

```shell
System:
    python: 3.11.7
Python dependencies:
      sklearn: 1.5.0
      pandas: 2.2.2
```
````

**Candidate Excerpt**

```
diff --git a/doc/whats_new/v1.5.rst b/doc/whats_new/v1.5.rst
index 20184bbd2a551..059875eec12d6 100644
--- a/doc/whats_new/v1.5.rst
+++ b/doc/whats_new/v1.5.rst
@@ -13,6 +13,23 @@ For a short description of the main highlights of the release, please refer to
 
 .. include:: changelog_legend.inc
 
+.. _changes_1_5_2:
+
+Version 1.5.2
+=============
+
+**release date of 1.5.2**
+
+Changelog
+---------
+
+:mod:`sklearn.compose`
+......................
+
+- |Fix| Fixed :class:`compose.TransformedTargetRegressor` not to raise `UserWarning` if
+  transform output is set to `pandas` or `polars`, since it isn't a transformer.
+  :pr:`29401` by :user:`Stefanie Senger <StefanieSenger>`.
+
 .. _changes_1_5_1:
 
 Version 1.5.1
diff --git a/sklearn/compose/_target.py b/sklearn/compose/_target.py
index ac33957b23ce2..db53eb9be9e65 100644
--- a/sklearn/compose/_target.py
+++ b/sklearn/compose/_target.py
@@ -198,6 +198,10 @@ def _fit_transformer(self, y):
                 validate=True,
                 check_inverse=self.check_inverse,
             )
+            # We are transforming the target here and not the features, so we set the
+            # output of FunctionTransformer() to be a numpy array (default) and to not
+            # depend on the global configuration:
+            self.transformer_.set_output(transform="default")
         # XXX: sample_weight is not currently passed to the
         # transformer. However, if transformer starts using sample_weight, the
         # code should be modified accordingly. At the time to consider the
diff --git a/sklearn/compose/tests/test_target.py b/sklearn/compose/tests/test_target.py
index a971553b64739..fd885459e76d1 100644
--- a/sklearn/compose/tests/test_target.py
+++ b/sklearn/compose/tests/test_target.py
@@ -1,7 +1,9 @@
+import w
...[truncated]
```

**Reviewer Decision To Fill In CSV**

- decision: `hard_accept` or `reject_or_ambiguous`
- reviewer: required
- reviewed_at: required UTC timestamp
- notes: required rationale

## 2. `nearmiss-v1:c-crab:0110087826d99378`

- Instance: `sympy__sympy-18410@189871b`
- Repo / PR: `sympy/sympy` / `18410`
- Candidate chars: `2343`
- Candidate redacted long digit runs: `False`
- Machine recommendation: `hard_accept_candidate`
- Machine reason: Candidate adds sinh and cosh to the recognized undetermined-coefficients function set and includes targeted dsolve/checkodesol coverage for the reported hyperbolic cases.
- Format concerns: none

**Task Excerpt**

````
Added hyperbolic function in undetermined coefficients

classify_ode doesn't recognise nth_linear_constant_coeff_undetermined_coefficients
> ```julia
> In [58]: eq = f(x).diff(x, 2) - 49*f(x) - sinh(3*x)                                                                                     
> 
> In [59]: classify_ode(eq)                                                                                                         
> Out[59]: 
> ('nth_linear_constant_coeff_variation_of_parameters', 
>  'nth_linear_constant_coeff_variation_of_parameters_Integral')
> 
> ```
Although `dsolve(eq)` gives the correct answer but it should also classify this equation as `nth_linear_constant_coeff_undetermined_coefficients`
If it could have identified then solution time will also reduce as 'nth_linear_constant_coeff_variation_of_parameters' is slower than `nth_linear_constant_coeff_undetermined_coefficients`
````

**Candidate Excerpt**

```
diff --git a/sympy/solvers/ode.py b/sympy/solvers/ode.py
index 2b6bca431c44..a41ba1851236 100644
--- a/sympy/solvers/ode.py
+++ b/sympy/solvers/ode.py
@@ -253,7 +253,7 @@
 
 from sympy.logic.boolalg import (BooleanAtom, And, Not, BooleanTrue,
                                 BooleanFalse)
-from sympy.functions import cos, exp, im, log, re, sin, tan, sqrt, \
+from sympy.functions import cos, cosh, exp, im, log, re, sin, sinh, tan, sqrt, \
     atan2, conjugate, Piecewise, cbrt, besselj, bessely, airyai, airybi
 from sympy.functions.combinatorial.factorials import factorial
 from sympy.integrals.integrals import Integral, integrate
@@ -5822,7 +5822,7 @@ def _test_term(expr, x):
                             foundtrig = True
             return all(_test_term(i, x) for i in expr.args)
         elif expr.is_Function:
-            if expr.func in (sin, cos, exp):
+            if expr.func in (sin, cos, exp, sinh, cosh):
                 if expr.args[0].match(a*x + b):
                     return True
                 else:
diff --git a/sympy/solvers/tests/test_ode.py b/sympy/solvers/tests/test_ode.py
index a4dc6160720f..4a608d84b24c 100644
--- a/sympy/solvers/tests/test_ode.py
+++ b/sympy/solvers/tests/test_ode.py
@@ -3830,3 +3830,20 @@ def test_issue_15996():
     sol = Eq(f(x), C1 + (C2 + C3*x - x**2/8 + 5*exp(2*I*x)/16)*sin(x) + (C4 + C5*x + I*x**2/8 + 5*I*exp(2*I*x)/16)*cos(x) - I*exp(I*x))
     assert sol == dsolve(eq, hint='nth_linear_constant_coeff_variation_of_parameters')
     assert checkodesol(eq, sol) == (True, 0)
+
+
+def test_issue_18408():
+    eq = f(x).diff(x, 3) - f(x).diff(x) - sinh(x)
+    sol = Eq(f(x), C1 + C2*exp(-x) + C3*exp(x) + x*sinh(x)/2)
+    assert sol == dsolve(eq, hint='nth_linear_constant_coeff_undetermined_coefficients')
+    assert checkodes
...[truncated]
```

**Reviewer Decision To Fill In CSV**

- decision: `hard_accept` or `reject_or_ambiguous`
- reviewer: required
- reviewed_at: required UTC timestamp
- notes: required rationale

## 3. `nearmiss-v1:c-crab:04b3b67ccf17ffe1`

- Instance: `python__mypy-3005@3587549`
- Repo / PR: `python/mypy` / `3005`
- Candidate chars: `13713`
- Candidate redacted long digit runs: `False`
- Machine recommendation: `hard_accept_candidate`
- Machine reason: Candidate implements issubclass-based Type narrowing, preserves unknown-type fallback behavior, and adds broad unit coverage for unions, tuples, builtins, and subclass hierarchies.
- Format concerns: Patch includes a submodule commit update; reviewer should ensure the packet preserves the intended submodule transition.

**Task Excerpt**

````
Infer types from issubclass() calls

Types should be promoted by issubclass
I expected types to be promoted to sub-types within an `issubclass` block, the same as instances within an `isinstance` block, but type-checking the following code results in the error `Type[Vehicle] has no attribute "wheels"`:

```python
from typing import Type

class Vehicle(object):
    name = ''

class Car(Vehicle):
    wheels = 4

def test_isinstance(x):
    # type: (Vehicle) -> None
    print(x.name)
    if isinstance(x, Car):
        print(x.wheels)

def test_issubclass(x):
    # type: (Type[Vehicle]) -> None
    print(x.name)
    if issubclass(x, Car):
        print(x.wheels)
```

Is this be possible or am I missing something?  I'm testing against the master branch.
````

**Candidate Excerpt**

```
diff --git a/mypy/checker.py b/mypy/checker.py
index 982ceca72576..488f3185e897 100644
--- a/mypy/checker.py
+++ b/mypy/checker.py
@@ -2675,6 +2675,21 @@ def or_conditional_maps(m1: TypeMap, m2: TypeMap) -> TypeMap:
     return result
 
 
+def convert_to_typetype(type_map: TypeMap) -> TypeMap:
+    converted_type_map = {}  # type: TypeMap
+    if type_map is None:
+        return None
+    for expr, typ in type_map.items():
+        if isinstance(typ, UnionType):
+            converted_type_map[expr] = UnionType([TypeType(t) for t in typ.items])
+        elif isinstance(typ, Instance):
+            converted_type_map[expr] = TypeType(typ)
+        else:
+            # unknown type; error was likely reported earlier
+            return {}
+    return converted_type_map
+
+
 def find_isinstance_check(node: Expression,
                           type_map: Dict[Expression, Type],
                           ) -> Tuple[TypeMap, TypeMap]:
@@ -2700,8 +2715,32 @@ def find_isinstance_check(node: Expression,
             expr = node.args[0]
             if expr.literal == LITERAL_TYPE:
                 vartype = type_map[expr]
-                types = get_isinstance_type(node.args[1], type_map)
-                return conditional_type_map(expr, vartype, types)
+                type = get_isinstance_type(node.args[1], type_map)
+                return conditional_type_map(expr, vartype, type)
+        elif refers_to_fullname(node.callee, 'builtins.issubclass'):
+            expr = node.args[0]
+            if expr.literal == LITERAL_TYPE:
+                vartype = type_map[expr]
+                type = get_isinstance_type(node.args[1], type_map)
+                if isinstance(vartype, UnionType):
+                    union_list = []
+                    for t in vartype.items:
+
...[truncated]
```

**Reviewer Decision To Fill In CSV**

- decision: `hard_accept` or `reject_or_ambiguous`
- reviewer: required
- reviewed_at: required UTC timestamp
- notes: required rationale

## 4. `nearmiss-v1:c-crab:04d390e945dd768e`

- Instance: `scikit-learn__scikit-learn-10099@33f7ffa`
- Repo / PR: `scikit-learn/scikit-learn` / `10099`
- Candidate chars: `3466`
- Candidate redacted long digit runs: `False`
- Machine recommendation: `hard_accept_candidate`
- Machine reason: Candidate emits a ConvergenceWarning when KMeans finds fewer distinct clusters than requested and adds a regression test matching duplicate-point behavior.
- Format concerns: none

**Task Excerpt**

````
[MRG+1] Warning in KMeans if too few clusters were found

Duplicated input points silently create duplicated clusters in KMeans
#### Description
When there are duplicated input points to Kmeans resulting to number of unique points < number of requested clusters, there is no error thrown. Instead, clustering continues to (seemingly) produce the number of clusters requested, but some of them are exactly the same, so the cluster labels produced for the input points do not go all the way to number of requested clusters.

#### Steps/Code to Reproduce
```python
from sklearn.cluster import KMeans
import numpy as np

# some input points here are identical, so that n_total=17, n_unique=9
x2d = np.array([(1086, 348), (1087, 347), (1190, 244), (1190, 244), (1086, 348), (1185, 249), (1193, 241), (1185, 249), (1087, 347), (1188, 247), (1187, 233), (26, 111), (26, 111), (26, 110), (26, 110), (26, 110), (26, 110)])
kmeans = KMeans(n_clusters=10) # n_clusters > n_unique
c_labels = kmeans.fit_predict(x2d)
c_centers = kmeans.cluster_centers_
```
#### Expected Results
Either an error thrown, or the cluster labels produced should match the unique clusters only (i.e. no identical cluster centres)

#### Actual Results
```python
>>> c_labels  # note there's no entry for cluster 9
array([7, 2, 6, 6, 7, 5, 4, 5, 2, 1, 3, 8, 8, 0, 0, 0, 0], dtype=int32)
>>> c_centers # two of these 10 clusters have identical centers, so only 9 of them are unique
array([[   26.,   110.],
       [ 1188.,   247.],
       [ 1087.,   347.],
       [ 1187.,   233.],
       [ 1193.,   241.],
       [ 1185.,   249.],
       [ 1190.,   244.],
       [ 1086.,   348.],
       [   26.,   111.],
       [   26.,   110.]]) 
```

#### Versions
```python
Darwin-16.7.0-x86_64-i386-64bit
Python
...[truncated]
````

**Candidate Excerpt**

```
diff --git a/doc/whats_new/v0.20.rst b/doc/whats_new/v0.20.rst
index 5af76499bcb39..de047f79a1924 100644
--- a/doc/whats_new/v0.20.rst
+++ b/doc/whats_new/v0.20.rst
@@ -134,6 +134,11 @@ Decomposition, manifold learning and clustering
 - Fixed a bug when setting parameters on meta-estimator, involving both a
   wrapped estimator and its parameter. :issue:`9999` by :user:`Marcus Voss
   <marcus-voss>` and `Joel Nothman`_.
+  
+- ``k_means`` now gives a warning, if the number of distinct clusters found
+  is smaller than ``n_clusters``. This may occur when the number of distinct 
+  points in the data set is actually smaller than the number of cluster one is 
+  looking for. :issue:`10059` by :user:`Christian Braune <christianbraune79>`.
 
 Metrics
 
diff --git a/sklearn/cluster/k_means_.py b/sklearn/cluster/k_means_.py
index 0da0144172703..fd31d5f84d94b 100644
--- a/sklearn/cluster/k_means_.py
+++ b/sklearn/cluster/k_means_.py
@@ -31,7 +31,7 @@
 from ..externals.joblib import Parallel
 from ..externals.joblib import delayed
 from ..externals.six import string_types
-
+from ..exceptions import ConvergenceWarning
 from . import _k_means
 from ._k_means_elkan import k_means_elkan
 
@@ -374,6 +374,13 @@ def k_means(X, n_clusters, init='k-means++', precompute_distances='auto',
             X += X_mean
         best_centers += X_mean
 
+    distinct_clusters = len(set(best_labels))
+    if distinct_clusters < n_clusters:
+        warnings.warn("Number of distinct clusters ({}) found smaller than "
+                      "n_clusters ({}). Possibly due to duplicate points "
+                      "in X.".format(distinct_clusters, n_clusters),
+                      ConvergenceWarning, stacklevel=2)
+
     if return_n_iter:
         return best_centers, best_labels, best_inertia,
...[truncated]
```

**Reviewer Decision To Fill In CSV**

- decision: `hard_accept` or `reject_or_ambiguous`
- reviewer: required
- reviewed_at: required UTC timestamp
- notes: required rationale

## 5. `nearmiss-v1:c-crab:060c7e12cfd0cb06`

- Instance: `dask__dask-10723@09b7b68`
- Repo / PR: `dask/dask` / `10723`
- Candidate chars: `12811`
- Candidate redacted long digit runs: `False`
- Machine recommendation: `hard_accept_candidate`
- Machine reason: Candidate is a focused pandas 3.0 migration-prep patch adding deprecation handling for affected dataframe, groupby, and rolling APIs with matching test updates.
- Format concerns: none

**Task Excerpt**

```
Pandas 3.0 deprecations and migration prep for dask-expr

DEPR: pandas 3.0 deprecations and the migration to dask-expr
pandas deprecated a bunch of things in the 2.x series that will be removed in pandas 3.0. Some of these things are changes in behaviour that are hard to replace but other things are just syntactic sugar. Examples like this are:

- [ ] GroupBy.fillna -> you can either use ffill/bfill or a regular fillna without a GroupBy at all, this method is useless
- [ ] GroupBys with axis=1 -> no need for grouping at all
- [ ] Rolling axis 1
- [ ] DataFrame.first/last
- [ ] axis in GroupBy.idxmin GroupBy.idxmax
- [ ] axis in GroupBy.shift

I don't want to implement these things for dask-expr just to remove them in a couple of months again, so I propose to deprecate this directly in dask/dask so that we can enforce it when we switch over to dask-expr
```

**Candidate Excerpt**

```
diff --git a/dask/dataframe/core.py b/dask/dataframe/core.py
index 7cddc42d933..03d4512251f 100644
--- a/dask/dataframe/core.py
+++ b/dask/dataframe/core.py
@@ -3735,6 +3735,12 @@ def resample(self, rule, closed=None, label=None):
 
         return Resampler(self, rule, closed=closed, label=label)
 
+    @_deprecated(
+        message=(
+            "Will be removed in a future version. "
+            "Please create a mask and filter using .loc instead"
+        )
+    )
     @derived_from(pd.DataFrame)
     def first(self, offset):
         # Let pandas error on bad args
@@ -3769,6 +3775,12 @@ def first(self, offset):
         graph = HighLevelGraph.from_collections(name, dsk, dependencies=[self])
         return new_dd_object(graph, name, self, divs)
 
+    @_deprecated(
+        message=(
+            "Will be removed in a future version. "
+            "Please create a mask and filter using .loc instead"
+        )
+    )
     @derived_from(pd.DataFrame)
     def last(self, offset):
         # Let pandas error on bad args
diff --git a/dask/dataframe/groupby.py b/dask/dataframe/groupby.py
index 955f19191a2..6220b294e22 100644
--- a/dask/dataframe/groupby.py
+++ b/dask/dataframe/groupby.py
@@ -1895,6 +1895,11 @@ def idxmin(
         skipna=True,
         numeric_only=no_default,
     ):
+        if axis != no_default:
+            warnings.warn(
+                "`axis` parameter is deprecated and will be removed in a future version.",
+                FutureWarning,
+            )
         if axis in (1, "columns"):
             raise NotImplementedError(
                 f"The axis={axis} keyword is not implemented for groupby.idxmin"
@@ -1925,6 +1930,12 @@ def idxmax(
         skipna=True,
         numeric_only=no_default,
     ):
+        if axis != no_default:
+
...[truncated]
```

**Reviewer Decision To Fill In CSV**

- decision: `hard_accept` or `reject_or_ambiguous`
- reviewer: required
- reviewed_at: required UTC timestamp
- notes: required rationale

## 6. `nearmiss-v1:c-crab:0735945503ef9330`

- Instance: `ManimCommunity__manim-509@1cbc24f`
- Repo / PR: `ManimCommunity/manim` / `509`
- Candidate chars: `2288`
- Candidate redacted long digit runs: `False`
- Machine recommendation: `hard_accept_candidate`
- Machine reason: Candidate adds ValueTracker in-place add support through __iadd__, preserves regular increment_value behavior, and covers the new += path with a test and changelog entry.
- Format concerns: none

**Task Excerpt**

````
FEAT : Add increment using += support for ValueTracker

ValueTracker must support increment also using +=
Just what the title says. Here's an example 
```python
class Test(Scene):
    def construct(self):
        tracker = ValueTracker(0.0)
        tracker.set_value(2.0)
        print(tracker.get_value()) # -> 2.000
        tracker.increment_value(3.0)
        print(tracker.get_value()) # -> 5.000
        tracker += 4.0 # NEW proposed feature
        print(tracker.get_value()) # -> 9.000
```
````

**Candidate Excerpt**

```
diff --git a/docs/source/changelog.rst b/docs/source/changelog.rst
index 940c8efa89..6d410549c1 100644
--- a/docs/source/changelog.rst
+++ b/docs/source/changelog.rst
@@ -62,6 +62,7 @@ Mobjects, Scenes, and Animations
 #. Add a :code:`Variable` class for displaying text that continuously updates to reflect the value of a python variable.
 #. The ``Tex`` and ``MathTex`` objects allow you to specify a custom TexTemplate using the ``template`` keyword argument.
 #. :code:`VGroup` now supports printing the class names of contained mobjects and :code:`VDict` supports printing the internal dict of mobjects
+#. :code:`ValueTracker` now supports increment using the `+=` operator (in addition to the already existing `increment_value` method)
 
 
 Documentation
diff --git a/manim/mobject/value_tracker.py b/manim/mobject/value_tracker.py
index 13ae1d7586..5cc2c3f790 100644
--- a/manim/mobject/value_tracker.py
+++ b/manim/mobject/value_tracker.py
@@ -57,6 +57,10 @@ def set_value(self, value):
     def increment_value(self, d_value):
         self.set_value(self.get_value() + d_value)
 
+    def __iadd__(self, d_value):
+        self.increment_value(d_value)
+        return self
+
     def interpolate(self, mobject1, mobject2, alpha, path_func=straight_path):
         """
         Turns self into an interpolation between mobject1
diff --git a/tests/test_value_tracker.py b/tests/test_value_tracker.py
new file mode 100644
index 0000000000..5f6c5bd69b
--- /dev/null
+++ b/tests/test_value_tracker.py
@@ -0,0 +1,28 @@
+from manim.mobject.value_tracker import ValueTracker
+
+
+def test_value_tracker_set_value():
+    """Test ValueTracker.set_value()"""
+    tracker = ValueTracker()
+    tracker.set_value(10.0)
+    assert tracker.get_value() == 10.0
+
+
+def test_value_tracker_get_value():
...[truncated]
```

**Reviewer Decision To Fill In CSV**

- decision: `hard_accept` or `reject_or_ambiguous`
- reviewer: required
- reviewed_at: required UTC timestamp
- notes: required rationale

## 7. `nearmiss-v1:c-crab:08cafcb6483d8389`

- Instance: `pandas-dev__pandas-33962@fc4993f`
- Repo / PR: `pandas-dev/pandas` / `33962`
- Candidate chars: `8718`
- Candidate redacted long digit runs: `True`
- Machine recommendation: `hard_accept_candidate`
- Machine reason: Candidate adds nrows plumbing and validation for line-delimited JSON reads, handles chunked and non-chunked paths, and includes tests and docs for the new behavior.
- Format concerns: candidate_redacted_long_digit_runs is true; visible redaction appears compatible with diff index metadata rather than substantive code.

**Task Excerpt**

````
Add nrows to read json.

ENH: Add nrows parameter to pd.read_json
#### Is your feature request related to a problem?

Let's say I have a huge `jsonlines` file and I want to read only the first `n` lines of the file.

#### Describe the solution you'd like

This problem could be fixed by adding `nrows` parameter in method `pd.read_json`. This parameter should be applicable if and only if `lines=True`

#### API breaking implications

Simply add and implement `nrows` parameter in `pd.read_json` (like `pd.read_csv`).

#### Additional context

How this enhancement could work:

```python
import pandas as pd

df = pd.read_json('/path/to/file.jsonlines', lines=True, nrows=1000)
```
````

**Candidate Excerpt**

```
diff --git a/asv_bench/benchmarks/io/json.py b/asv_bench/benchmarks/io/json.py
index f478bf2aee0ba..a490e250943f5 100644
--- a/asv_bench/benchmarks/io/json.py
+++ b/asv_bench/benchmarks/io/json.py
@@ -53,12 +53,18 @@ def time_read_json_lines(self, index):
     def time_read_json_lines_concat(self, index):
         concat(read_json(self.fname, orient="records", lines=True, chunksize=25000))
 
+    def time_read_json_lines_nrows(self, index):
+        read_json(self.fname, orient="records", lines=True, nrows=25000)
+
     def peakmem_read_json_lines(self, index):
         read_json(self.fname, orient="records", lines=True)
 
     def peakmem_read_json_lines_concat(self, index):
         concat(read_json(self.fname, orient="records", lines=True, chunksize=25000))
 
+    def peakmem_read_json_lines_nrows(self, index):
+        read_json(self.fname, orient="records", lines=True, nrows=15000)
+
 
 class ToJSON(BaseIO):
 
diff --git a/doc/source/whatsnew/v1.1.0.rst b/doc/source/whatsnew/v1.1.0.rst
index 17623b943bf87..2243790a663df 100644
--- a/doc/source/whatsnew/v1.1.0.rst
+++ b/doc/source/whatsnew/v1.1.0.rst
@@ -289,6 +289,7 @@ Other enhancements
 - Make :class:`pandas.core.window.Rolling` and :class:`pandas.core.window.Expanding` iterable（:issue:`11704`)
 - Make ``option_context`` a :class:`contextlib.ContextDecorator`, which allows it to be used as a decorator over an entire function (:issue:`34253`).
 - :meth:`groupby.transform` now allows ``func`` to be ``pad``, ``backfill`` and ``cumcount`` (:issue:`31269`).
+- :meth:`~pandas.io.json.read_json` now accepts `nrows` parameter. (:issue:`33916`).
 - :meth `~pandas.io.gbq.read_gbq` now allows to disable progress bar (:issue:`33360`).
 
 .. ---------------------------------------------------------------------------
diff --gi
...[truncated]
```

**Reviewer Decision To Fill In CSV**

- decision: `hard_accept` or `reject_or_ambiguous`
- reviewer: required
- reviewed_at: required UTC timestamp
- notes: required rationale

## 8. `nearmiss-v1:c-crab:09584d0209952576`

- Instance: `numpy__numpy-14924@6a7e3f1`
- Repo / PR: `numpy/numpy` / `14924`
- Candidate chars: `6927`
- Candidate redacted long digit runs: `True`
- Machine recommendation: `hard_accept_candidate`
- Machine reason: Candidate replaces the small-alpha Dirichlet failure mode with a stick-breaking path, keeps the standard gamma-normalization path for normal cases, and adds small-alpha tests plus release documentation.
- Format concerns: candidate_redacted_long_digit_runs is true; visible redaction appears limited to diff index metadata.

**Task Excerpt**

````
BUG: Fix numpy.random.dirichlet returns NaN for small 'alpha' parameters.

Bug in np.random.dirichlet for small alpha parameters
Hi,

I encountered a bug when using np.random.dirichlet with small alpha parameters. Call and traceback are below.

``` python
ZeroDivisionError                         Traceback (most recent call last)
<ipython-input-86-73c2067e20c1> in <module>()
----> 1 np.random.dirichlet([0.0001, 0.0, 0.0001])

mtrand.pyx in mtrand.RandomState.dirichlet (numpy/random/mtrand/mtrand.c:24477)()

mtrand.pyx in mtrand.RandomState.dirichlet (numpy/random/mtrand/mtrand.c:24387)()

ZeroDivisionError: float division
```

I am using numpy-1.9.1. 

I believe this is a floating point issue, the distribution has almost all of its mass very close to either (1, 0, 0) or (0, 0, 1) .

The 'float division' error already occurs for larger values<1, e.g. 0.001. 

It is likely that this occurs because of the Dirichlet distribution is usually sampled via the Gamma distribution followed by normalization. If all values returned from Gamma sampling are zero than a float division error occurs. 

In addition 

``` python
np.random.beta(0.0001, 0.0001)
```

produces 'nan' most of the time, while it should be alternating 'almost always' between (1, 0) and 
(0, 1) 

`python scipy.special.betainc(0.0001, 0.0001, 1e-50) = 0.494..`

It might not be able to fix that in the current algorithmic framework but maybe it is possible to discourage/prevent users from supplying too small parameters.

Wow, this wasn't supposed to become such a long post. Thanks to anyone reading/considering this issue.
````

**Candidate Excerpt**

```
diff --git a/doc/release/upcoming_changes/14924.compatibility.rst b/doc/release/upcoming_changes/14924.compatibility.rst
new file mode 100644
index [redacted-long-digit-run]..8b42437fd502
--- /dev/null
+++ b/doc/release/upcoming_changes/14924.compatibility.rst
@@ -0,0 +1,7 @@
+Changed random variate stream from `numpy.random.Generator.dirichlet`
+---------------------------------------------------------------------
+A bug in the generation of random variates for the Dirichlet distribution
+with small `alpha` values was fixed by using a different algorithm when
+``max(alpha) < 0.1``.  Because of the change, the stream of variates
+generated by `dirichlet` in this case will be different from previous
+releases.
diff --git a/numpy/random/_generator.pyx b/numpy/random/_generator.pyx
index b6c222cc0720..4385cb698e9d 100644
--- a/numpy/random/_generator.pyx
+++ b/numpy/random/_generator.pyx
@@ -4124,10 +4124,12 @@ cdef class Generator:
         # return val
 
         cdef np.npy_intp k, totsize, i, j
-        cdef np.ndarray alpha_arr, val_arr
+        cdef np.ndarray alpha_arr, val_arr, alpha_csum_arr
+        cdef double csum
         cdef double *alpha_data
+        cdef double *alpha_csum_data
         cdef double *val_data
-        cdef double acc, invacc
+        cdef double acc, invacc, v
 
         k = len(alpha)
         alpha_arr = <np.ndarray>np.PyArray_FROM_OTF(
@@ -4150,17 +4152,74 @@ cdef class Generator:
 
         i = 0
         totsize = np.PyArray_SIZE(val_arr)
-        with self.lock, nogil:
-            while i < totsize:
-                acc = 0.0
-                for j in range(k):
-                    val_data[i+j] = random_standard_gamma(&self._bitgen,
-                                                              alpha_data[j])
-                    a
...[truncated]
```

**Reviewer Decision To Fill In CSV**

- decision: `hard_accept` or `reject_or_ambiguous`
- reviewer: required
- reviewed_at: required UTC timestamp
- notes: required rationale

## 9. `nearmiss-v1:c-crab:0b5adcf2e8a30f49`

- Instance: `scikit-learn__scikit-learn-23548@cc5e175`
- Repo / PR: `scikit-learn/scikit-learn` / `23548`
- Candidate chars: `2844`
- Candidate redacted long digit runs: `False`
- Machine recommendation: `hard_accept_candidate`
- Machine reason: Candidate changes coverage_error input validation to require 2D arrays and adds regression coverage for the expected 1D error message and common invalid-input tests.
- Format concerns: none

**Task Excerpt**

````
FIX Ensure correct sklearn.metrics.coverage_error error message for 1D array

sklearn.metrics.coverage_error wrong error message for 1D array
### Describe the bug

Let y_true and y_score be numpy arrays of shape (K,),
when you try to run the "sklearn.metrics.coverage_error" as explained in the documentation
it returns "binary type not supported" error, but this is not the case at all, the metric works with,
binary, the problem is it expects matrices of shape (1, K) if there is only one sample,
Please either fix the function so that it can work with 1D array or please fix the error message.

### Steps/Code to Reproduce

```python
coverage = np.mean([coverage_error(discrete_labels[i], discrete_scores[i]) - 1 for i in range(N)])
```
### Expected Results

No error or, can not use 1D array error

### Actual Results

```python-traceback
   raise ValueError("{0} format is not supported".format(y_type))
ValueError: binary format is not supported
```

### Versions

```shell
>>> import sklearn; sklearn.show_versions()

System:
    python: 3.9.7 | packaged by conda-forge | (default, Sep 29 2021, 19:20:46)  [GCC 9.4.0]
executable: /home/delivahit/anaconda3/envs/bugra/bin/python
   machine: Linux-5.4.0-107-generic-x86_64-with-glibc2.27

Python dependencies:
          pip: 21.2.4
   setuptools: 58.0.4
      sklearn: 1.0.2
        numpy: 1.21.2
        scipy: 1.7.2
       Cython: None
       pandas: 1.3.4
   matplotlib: 3.5.0
       joblib: 1.1.0
threadpoolctl: 3.0.0

Built with OpenMP: True
```
````

**Candidate Excerpt**

```
diff --git a/doc/whats_new/v1.2.rst b/doc/whats_new/v1.2.rst
index ede154619526e..d93c7bb698ea6 100644
--- a/doc/whats_new/v1.2.rst
+++ b/doc/whats_new/v1.2.rst
@@ -119,6 +119,9 @@ Changelog
   of a binary classification problem. :pr:`22518` by
   :user:`Arturo Amor <ArturoAmorQ>`.
 
+- |Fix| Fixed error message of :class:`metrics.coverage_error` for 1D array input.
+  :pr:`23548` by :user:`Hao Chun Chang <haochunchang>`.
+
 :mod:`sklearn.neighbors`
 ........................
 
diff --git a/sklearn/metrics/_ranking.py b/sklearn/metrics/_ranking.py
index 7f64f479ed275..9e746f167381f 100644
--- a/sklearn/metrics/_ranking.py
+++ b/sklearn/metrics/_ranking.py
@@ -1155,8 +1155,8 @@ def coverage_error(y_true, y_score, *, sample_weight=None):
            handbook (pp. 667-685). Springer US.
 
     """
-    y_true = check_array(y_true, ensure_2d=False)
-    y_score = check_array(y_score, ensure_2d=False)
+    y_true = check_array(y_true, ensure_2d=True)
+    y_score = check_array(y_score, ensure_2d=True)
     check_consistent_length(y_true, y_score, sample_weight)
 
     y_type = type_of_target(y_true, input_name="y_true")
diff --git a/sklearn/metrics/tests/test_common.py b/sklearn/metrics/tests/test_common.py
index 1e627f9f86676..c0d6d351b8c3e 100644
--- a/sklearn/metrics/tests/test_common.py
+++ b/sklearn/metrics/tests/test_common.py
@@ -910,6 +910,10 @@ def test_thresholded_invariance_string_vs_numbers_labels(name):
 )
 @pytest.mark.parametrize("y_true, y_score", invalids_nan_inf)
 def test_regression_thresholded_inf_nan_input(metric, y_true, y_score):
+    # Reshape since coverage_error only accepts 2D arrays.
+    if metric == coverage_error:
+        y_true = [y_true]
+        y_score = [y_score]
     with pytest.raises(ValueError, match=r"contains (NaN|infinity)"):
...[truncated]
```

**Reviewer Decision To Fill In CSV**

- decision: `hard_accept` or `reject_or_ambiguous`
- reviewer: required
- reviewed_at: required UTC timestamp
- notes: required rationale

## 10. `nearmiss-v1:c-crab:0c6318021a8a500b`

- Instance: `pandas-dev__pandas-24968@0ea6653`
- Repo / PR: `pandas-dev/pandas` / `24968`
- Candidate chars: `6266`
- Candidate redacted long digit runs: `False`
- Machine recommendation: `hard_accept_candidate`
- Machine reason: Candidate exposes symlog scale through the pandas plotting API and adds DataFrame and Series plotting tests plus documentation.
- Format concerns: none

**Task Excerpt**

````
ENH: Expose symlog scaling in plotting API

ENH: Expose symlog scaling in plotting API
The default log-scaled axes, activated by the `logx`, `logy`, and `loglog` methods to the Pandas plotting API, do the straightforward thing and take the log of 0 values. It then attempt to plot with these infinite logs, and makes the entire plot unusable without warning in the presence of 0s.

For example:

```python
draws = pd.DataFrame({'freq': np.random.zipf(1.7, 1000) - 1})
draws['rank'] = (-draws['freq']).rank()
draws.plot(x='rank', y='freq', kind='scatter', loglog=True)
```

Matplotlib provides another scale, the `symlog` scale, that makes a small region near 0 linear to avoid these problems. For quick-and-dirty 'look at my data on a log axis' plotting, `symlog` is significantly more useful.

I can access it like this:

```
draws = pd.DataFrame({'freq': np.random.zipf(1.7, 1000) - 1})
draws['rank'] = (-draws['freq']).rank()
p = draws.plot(x='rank', y='freq', kind='scatter', loglog=True)
p.set_xscale('symlog')
p.set_yscale('symlog')
p
```

Either making the `symlog` scale the default log scale for plotting, or supporting a `loglog='sym'` option, would make it significantly easier to do quick data inspection with Pandas' convenience plotting.
````

**Candidate Excerpt**

```
diff --git a/doc/source/whatsnew/v0.25.0.rst b/doc/source/whatsnew/v0.25.0.rst
index ccf5c43280765..be208a434f77b 100644
--- a/doc/source/whatsnew/v0.25.0.rst
+++ b/doc/source/whatsnew/v0.25.0.rst
@@ -23,7 +23,7 @@ including other versions of pandas.
 
 Other Enhancements
 ^^^^^^^^^^^^^^^^^^
-
+- :func:`DataFrame.plot` keywords ``logy``, ``logx`` and ``loglog`` can now accept the value ``'sym'`` for symlog scaling. (:issue:`24867`)
 - Added support for ISO week year format ('%G-%V-%u') when parsing datetimes using :meth: `to_datetime` (:issue:`16607`)
 - Indexing of ``DataFrame`` and ``Series`` now accepts zerodim ``np.ndarray`` (:issue:`24919`)
 - :meth:`Timestamp.replace` now supports the ``fold`` argument to disambiguate DST transition times (:issue:`25017`)
diff --git a/pandas/plotting/_core.py b/pandas/plotting/_core.py
index 5ed6c2f4e14b6..e75e8bb4f8821 100644
--- a/pandas/plotting/_core.py
+++ b/pandas/plotting/_core.py
@@ -288,8 +288,10 @@ def _maybe_right_yaxis(self, ax, axes_num):
             if not self._has_plotted_object(orig_ax):  # no data on left y
                 orig_ax.get_yaxis().set_visible(False)
 
-            if self.logy or self.loglog:
+            if self.logy is True or self.loglog is True:
                 new_ax.set_yscale('log')
+            elif self.logy == 'sym' or self.loglog == 'sym':
+                new_ax.set_yscale('symlog')
             return new_ax
 
     def _setup_subplots(self):
@@ -311,10 +313,24 @@ def _setup_subplots(self):
 
         axes = _flatten(axes)
 
-        if self.logx or self.loglog:
+        valid_log = {False, True, 'sym', None}
+        input_log = {self.logx, self.logy, self.loglog}
+        if input_log - valid_log:
+            invalid_log = next(iter((input_log - valid_log)))
+            raise ValueE
...[truncated]
```

**Reviewer Decision To Fill In CSV**

- decision: `hard_accept` or `reject_or_ambiguous`
- reviewer: required
- reviewed_at: required UTC timestamp
- notes: required rationale

## 11. `nearmiss-v1:c-crab:0e31e881b1af8ab5`

- Instance: `reflex-dev__reflex-4406@5ec2c11`
- Repo / PR: `reflex-dev/reflex` / `4406`
- Candidate chars: `11317`
- Candidate redacted long digit runs: `False`
- Machine recommendation: `hard_accept_candidate`
- Machine reason: Candidate adds datetime Var comparison operations and serializer coverage, with integration tests exercising the new operations in the expected browser-facing workflow.
- Format concerns: none

**Task Excerpt**

````
add datetime var comparison operations

Unable to compare `date` objects
**Describe the bug**
I'm getting an error when trying to compare two `date` objects and one of them is a `State var` (see code below)

**To Reproduce**
Here's a code example
 ```
rx.flex(
      rx.cond(
          datetime.now(pytz.utc).date() <= competition.end_date,
          rx.button("Join", on_click=lambda: rx.redirect(f"/competitions/{competition.id}/join"), size="4"),
      ),
  ),
```
and the error I'm getting:
reflex.utils.exceptions.VarTypeError: Unsupported Operand type(s) for <: state__state.selected_competition?.end_date of type date and 2024-11-05 of type date

**Expected behavior**
No errors raised

**Screenshots**
Na

**Specifics (please complete the following information):**
 - Python Version: 3.12
 - Reflex Version: 0.5.2
 - OS: macOS Sonoma 14.5
 - Browser (Optional): Firefox
 
**Additional context**
Na
````

**Candidate Excerpt**

```
diff --git a/reflex/vars/__init__.py b/reflex/vars/__init__.py
index 1a4cebe19a3..cb02319bc6a 100644
--- a/reflex/vars/__init__.py
+++ b/reflex/vars/__init__.py
@@ -9,6 +9,7 @@
 from .base import get_uuid_string_var as get_uuid_string_var
 from .base import var_operation as var_operation
 from .base import var_operation_return as var_operation_return
+from .datetime import DateTimeVar as DateTimeVar
 from .function import FunctionStringVar as FunctionStringVar
 from .function import FunctionVar as FunctionVar
 from .function import VarOperationCall as VarOperationCall
diff --git a/reflex/vars/datetime.py b/reflex/vars/datetime.py
new file mode 100644
index 00000000000..a4548e6f732
--- /dev/null
+++ b/reflex/vars/datetime.py
@@ -0,0 +1,222 @@
+"""Immutable datetime and date vars."""
+
+from __future__ import annotations
+
+import dataclasses
+import sys
+from datetime import date, datetime
+from typing import Any, NoReturn, TypeVar, Union, overload
+
+from reflex.utils.exceptions import VarTypeError
+from reflex.vars.number import BooleanVar
+
+from .base import (
+    CustomVarOperationReturn,
+    LiteralVar,
+    Var,
+    VarData,
+    var_operation,
+    var_operation_return,
+)
+
+DATETIME_T = TypeVar("DATETIME_T", datetime, date)
+
+datetime_types = Union[datetime, date]
+
+
+def raise_var_type_error():
+    """Raise a VarTypeError.
+
+    Raises:
+        VarTypeError: Cannot compare a datetime object with a non-datetime object.
+    """
+    raise VarTypeError("Cannot compare a datetime object with a non-datetime object.")
+
+
+class DateTimeVar(Var[DATETIME_T], python_types=(datetime, date)):
+    """A variable that holds a datetime or date object."""
+
+    @overload
+    def __lt__(self, other: datetime_types) -> BooleanVar: ...
+
+    @overload
+    def __lt
...[truncated]
```

**Reviewer Decision To Fill In CSV**

- decision: `hard_accept` or `reject_or_ambiguous`
- reviewer: required
- reviewed_at: required UTC timestamp
- notes: required rationale

## 12. `nearmiss-v1:c-crab:0f20280ccef865cb`

- Instance: `home-assistant__core-142977@fa60db7`
- Repo / PR: `home-assistant/core` / `142977`
- Candidate chars: `3455`
- Candidate redacted long digit runs: `False`
- Machine recommendation: `hard_accept_candidate`
- Machine reason: Candidate applies enabled checks inside parallel script sequences so disabled nested sequence steps are skipped, with regression coverage for the reported automation/script behavior.
- Format concerns: none

**Task Excerpt**

````
Fix Automation/Script: sequence within a parallel ignoring enabled flag

Scripts/Automations - Sequence inside Parallel ignoring being disabled
### The problem

first noticed this in 2024.8 and remains in 2024.9.
running haos on a HA yellow

if you have a parallel building block that has sequence building blocks inside it then the sequences ignore "enabled: false" and run

so you have

parallel block
> Sequence Block 1
>>some stuff

> Sequence Block 2
>>some other stuff

even if "Sequence Block 2" is disabled then "some other stuff" still runs

have attached yaml for my repro script below using 2 dummy switch helpers and attached the trace showing both sequences running despite the 2nd one being disabled

Obviously a highly specific situation but caused me some issues that I had to work around so felt I should report

### What version of Home Assistant Core has the issue?

2024.9.1

### What was the last working version of Home Assistant Core?

_No response_

### What type of installation are you running?

Home Assistant OS

### Integration causing the issue

_No response_

### Link to integration documentation on our website

_No response_

### Diagnostics information

_No response_

### Example YAML snippet

```yaml
alias: bug repro
sequence:
  - parallel:
      - sequence:
          - action: input_boolean.toggle
            target:
              entity_id: input_boolean.dummy_switch_1
            data: {}
      - sequence:
          - action: input_boolean.toggle
            target:
              entity_id: input_boolean.dummy_switch_2
            data: {}
            enabled: true
        enabled: false
description: ""
```


### Anything in the logs that might be useful for us?

```txt
{
  "tra
...[truncated]
````

**Candidate Excerpt**

```
diff --git a/homeassistant/helpers/script.py b/homeassistant/helpers/script.py
index 43429bdb1d26d..2b4da38b15efb 100644
--- a/homeassistant/helpers/script.py
+++ b/homeassistant/helpers/script.py
@@ -629,6 +629,10 @@ async def _async_run_script(
         self, script: Script, *, parallel: bool = False
     ) -> None:
         """Execute a script."""
+        if not script.enabled:
+            self._log("Skipping disabled script: %s", script.name)
+            trace_set_result(enabled=False)
+            return
         result = await self._async_run_long_action(
             self._hass.async_create_task_internal(
                 script.async_run(
@@ -1442,8 +1446,12 @@ def __init__(
         script_mode: str = DEFAULT_SCRIPT_MODE,
         top_level: bool = True,
         variables: ScriptVariables | None = None,
+        enabled: bool = True,
     ) -> None:
-        """Initialize the script."""
+        """Initialize the script.
+
+        enabled attribute is only used for non-top-level scripts.
+        """
         if not (all_scripts := hass.data.get(DATA_SCRIPTS)):
             all_scripts = hass.data[DATA_SCRIPTS] = []
             hass.bus.async_listen_once(
@@ -1462,6 +1470,7 @@ def __init__(
         self.name = name
         self.unique_id = f"{domain}.{name}-{id(self)}"
         self.domain = domain
+        self.enabled = enabled
         self.running_description = running_description or f"{domain} script"
         self._change_listener = change_listener
         self._change_listener_job = (
@@ -2002,6 +2011,7 @@ async def _async_prep_parallel_scripts(self, step: int) -> list[Script]:
                 max_runs=self.max_runs,
                 logger=self._logger,
                 top_level=False,
+                enabled=parallel_script.get(CONF_ENABLED
...[truncated]
```

**Reviewer Decision To Fill In CSV**

- decision: `hard_accept` or `reject_or_ambiguous`
- reviewer: required
- reviewed_at: required UTC timestamp
- notes: required rationale

## 13. `nearmiss-v1:c-crab:10070430d41b73e9`

- Instance: `sympy__sympy-24666@7d194c7`
- Repo / PR: `sympy/sympy` / `24666`
- Candidate chars: `2064`
- Candidate redacted long digit runs: `True`
- Machine recommendation: `hard_accept_candidate`
- Machine reason: Candidate scope is explicitly a regression test for simplified Piecewise condition preservation; the patch adds focused test coverage matching that test-only task.
- Format concerns: candidate_redacted_long_digit_runs is true; reviewer should confirm redaction does not hide a significant literal in the expected Piecewise expression.

**Task Excerpt**

````
Added test for simplified Piecewise is missing conditions

simplified Piecewise is missing conditions
I have been trying to write a model for the size of symbolic powers.
I define the Piecewise expression as follows for power `b**e` with
the expressions of the Piecewise giving a value that is representative
of the region in which the size of the power is found (e.g. 1/2 if the
value of the power is between 0 and 1):

```python
var('b e')
A=Piecewise((1, Eq(b, 1) | Eq(e, 0) | (Eq(b, -1) & Eq(Mod(e, 2), 0))),
(0, Eq(b, 0) & (e > 0)), (-1, Eq(b, -1) & Eq(Mod(e, 2), 1)),
(Piecewise((2, ((b > 1) & (e > 0)) | ((b > 0) & (b < 1) & (e < 0)) |
((e >= 2) & (b < -1) & Eq(Mod(e, 2), 0)) | ((e <= -2) & (b > -1) & (b
< 0) & Eq(Mod(e, 2), 0))), (S.Half, ((b > 1) & (e < 0)) | ((b > 0) & (e >
0) & (b < 1)) | ((e <= -2) & (b < -1) & Eq(Mod(e, 2), 0)) | ((e >= 2)
& (b > -1) & (b < 0) & Eq(Mod(e, 2), 0))), (-S.Half, Eq(Mod(e, 2), 1) &
(((e <= -1) & (b < -1)) | ((e >= 1) & (b > -1) & (b < 0)))), (-2, ((e
>= 1) & (b < -1) & Eq(Mod(e, 2), 1)) | ((e <= -1) & (b > -1) & (b < 0)
& Eq(Mod(e, 2), 1)))), Eq(im(b), 0) & Eq(im(e), 0)))
```
As a test of folding and simplification I create the following alternative forms for A
```python
B=piecewise_fold(A)
sa=A.simplify()
sb=B.simplify()
```
I then test the forms over a range of `b` and `e` values:
```python
v = Tuple(-2, -1, -0.5, 0, 0.5, 1, 2)
for i in v:
 for j in v:
  r = {b:i,e:j}
  ok=[k.xreplace(r) for k in (A,B,sa,sb)]
  if len(set(ok))!=1:print('ab %s %s'%(r,ok))
```
Although `A` and `B` agree, the simplified forms of each do not and I get these results
```python
s {b: -2, e: -1} [-1/2, -1/2, -1/2, nan]
s {b: -2, e: 1} [-2, -2, -2, nan]
s {b: -1/2, e: 1} [-1/2, -1/2, -1/2, nan]
```
The `nan` indicates th
...[truncated]
````

**Candidate Excerpt**

```
diff --git a/sympy/functions/elementary/tests/test_piecewise.py b/sympy/functions/elementary/tests/test_piecewise.py
index 958c671c470f..2d4de12b284e 100644
--- a/sympy/functions/elementary/tests/test_piecewise.py
+++ b/sympy/functions/elementary/tests/test_piecewise.py
@@ -5,6 +5,7 @@
 from sympy.core.expr import unchanged
 from sympy.core.function import (Function, diff, expand)
 from sympy.core.mul import Mul
+from sympy.core.mod import Mod
 from sympy.core.numbers import (Float, I, Rational, oo, pi, zoo)
 from sympy.core.relational import (Eq, Ge, Gt, Ne)
 from sympy.core.singleton import S
@@ -1339,6 +1340,43 @@ def test_issue_14787():
     f = Piecewise((x, x < 1), ((S(58) / 7), True))
     assert str(f.evalf()) == "Piecewise((x, x < 1), (8.[redacted-long-digit-run], True))"
 
+def test_issue_21481():
+    b, e = symbols('b e')
+    C = Piecewise(
+        (2,
+        ((b > 1) & (e > 0)) |
+        ((b > 0) & (b < 1) & (e < 0)) |
+        ((e >= 2) & (b < -1) & Eq(Mod(e, 2), 0)) |
+        ((e <= -2) & (b > -1) & (b < 0) & Eq(Mod(e, 2), 0))),
+        (S.Half,
+        ((b > 1) & (e < 0)) |
+        ((b > 0) & (e > 0) & (b < 1)) |
+        ((e <= -2) & (b < -1) & Eq(Mod(e, 2), 0)) |
+        ((e >= 2) & (b > -1) & (b < 0) & Eq(Mod(e, 2), 0))),
+        (-S.Half,
+        Eq(Mod(e, 2), 1) &
+        (((e <= -1) & (b < -1)) | ((e >= 1) & (b > -1) & (b < 0)))),
+        (-2,
+        ((e >= 1) & (b < -1) & Eq(Mod(e, 2), 1)) |
+        ((e <= -1) & (b > -1) & (b < 0) & Eq(Mod(e, 2), 1)))
+    )
+    A = Piecewise(
+        (1, Eq(b, 1) | Eq(e, 0) | (Eq(b, -1) & Eq(Mod(e, 2), 0))),
+        (0, Eq(b, 0) & (e > 0)),
+        (-1, Eq(b, -1) & Eq(Mod(e, 2), 1)),
+        (C, Eq(im(b), 0) & Eq(im(e), 0))
+    )
+
+    B = piecewise_fold(A)
+    sa = A.simplify()
+    sb =
...[truncated]
```

**Reviewer Decision To Fill In CSV**

- decision: `hard_accept` or `reject_or_ambiguous`
- reviewer: required
- reviewed_at: required UTC timestamp
- notes: required rationale

## 14. `nearmiss-v1:c-crab:12dedf2f36029e2c`

- Instance: `scikit-learn__scikit-learn-29442@3a19575`
- Repo / PR: `scikit-learn/scikit-learn` / `29442`
- Candidate chars: `11363`
- Candidate redacted long digit runs: `False`
- Machine recommendation: `hard_accept_candidate`
- Machine reason: Candidate fixes ElasticNetCV sample-weight handling in coordinate descent and includes targeted linear-model regression tests plus release documentation.
- Format concerns: none

**Task Excerpt**

````
Fix elasticnect cv sample weight

Calculation of alphas in ElasticNetCV doesn't use sample_weight
### Describe the bug

In ElasticNetCV, the first and largest value of `alpha`, call it `alpha_max`, should be just large enough to force all of the coefficients to become zero. The existing code works correctly when `sample_weight` is not specified. However, the computation of `alpha_max` does not take into account `sample_weight`.

### Steps/Code to Reproduce

```python
import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV

X = np.array([[3, 1], [2, 5], [5, 3], [1, 4]])
beta = np.array([1, 1])
y = X @ beta
w = np.array([10, 1, 10, 1])

# Fit ElasticNetCV just to get the .alphas_ attribute
enetCV = ElasticNetCV(cv=2)
enetCV.fit(X, y, sample_weight=w)

# The coefficient of ElasticNet fitted at alpha_max should be [0.  0.].
alpha_max = enetCV.alphas_[0]
enet = ElasticNet(alpha=alpha_max)
enet.fit(X, y, sample_weight=w)
print(enet.coef_)  # [0.1970807  0.19708023]
```

### Expected Results

If the correct value of `alpha_max` is computed, then `enet.coef_` should be right at the cusp of zero, such that any smaller value of `alpha` makes it nonzero:
```python
def get_alpha_max(X, y, w, l1_ratio=0.5):
    wn = w / w.sum()
    Xn = X - np.dot(wn, X)
    yn = (y - np.dot(wn, y)) * wn
    return np.max(np.abs(yn @ Xn)) / l1_ratio


enet = ElasticNet(alpha=get_alpha_max(X, y, w))
enet.fit(X, y, sample_weight=w)
print(enet.coef_)  # [6.70427878e-17 6.70427878e-17]
```

### Actual Results

`enet.coef_` is `[0.1970807  0.19708023]`.

### Versions

```shell
System:
    python: 3.9.7 (default, Sep 16 2021, 13:09:58)  [GCC 7.5.0]
executable: /home/jhopfens/.conda/envs/jhop39/bin/python
   machine: Linux-3.10.0-1160
...[truncated]
````

**Candidate Excerpt**

```
diff --git a/doc/whats_new/v1.6.rst b/doc/whats_new/v1.6.rst
index a45765cbd04a3..5d56e534364fd 100644
--- a/doc/whats_new/v1.6.rst
+++ b/doc/whats_new/v1.6.rst
@@ -233,6 +233,11 @@ Changelog
   has no effect. `copy_X` will be removed in 1.8.
   :pr:`29105` by :user:`Adam Li <adam2392>`.
 
+- |Fix| :class:`linear_model.LassoCV` and :class:`linear_model.ElasticNetCV` now
+  take sample weights into accounts to define the search grid for the internally tuned
+  `alpha` hyper-parameter. :pr:`29442` by :user:`John Hopfensperger <s-banach> and
+  :user:`Shruti Nath <snath-xoc>`.
+
 :mod:`sklearn.manifold`
 .......................
 
diff --git a/sklearn/linear_model/_coordinate_descent.py b/sklearn/linear_model/_coordinate_descent.py
index 4ede6c04b462b..0c1421ebffe89 100644
--- a/sklearn/linear_model/_coordinate_descent.py
+++ b/sklearn/linear_model/_coordinate_descent.py
@@ -98,6 +98,7 @@ def _alpha_grid(
     eps=1e-3,
     n_alphas=100,
     copy_X=True,
+    sample_weight=None,
 ):
     """Compute the grid of alpha values for elastic net parameter search
 
@@ -132,6 +133,8 @@ def _alpha_grid(
 
     copy_X : bool, default=True
         If ``True``, X will be copied; else, it may be overwritten.
+
+    sample_weight : ndarray of shape (n_samples,), default=None
     """
     if l1_ratio == 0:
         raise ValueError(
@@ -140,43 +143,39 @@ def _alpha_grid(
             "your estimator with the appropriate `alphas=` "
             "argument."
         )
-    n_samples = len(y)
-
-    sparse_center = False
-    if Xy is None:
-        X_sparse = sparse.issparse(X)
-        sparse_center = X_sparse and fit_intercept
-        X = check_array(
-            X, accept_sparse="csc", copy=(copy_X and fit_intercept and not X_sparse)
+    if Xy is not None:
+        Xyw = Xy
+    e
...[truncated]
```

**Reviewer Decision To Fill In CSV**

- decision: `hard_accept` or `reject_or_ambiguous`
- reviewer: required
- reviewed_at: required UTC timestamp
- notes: required rationale

## 15. `nearmiss-v1:c-crab:1600ca8239e2f6e0`

- Instance: `python__mypy-11143@23a0e4b`
- Repo / PR: `python/mypy` / `11143`
- Candidate chars: `14542`
- Candidate redacted long digit runs: `True`
- Machine recommendation: `hard_accept_candidate`
- Machine reason: Candidate expands mypy package discovery to search sys.path for PEP-561 compliant packages and adds command-line/modulefinder tests plus fixture package path cases.
- Format concerns: candidate_redacted_long_digit_runs is true; redaction may affect path or fixture literal review and should be checked manually before authoritative signoff.

**Task Excerpt**

```
Search sys.path for PEP-561 compliant packages

sitecustomize/full sys.path  support
Looking at sitepkgs.py, the current implementation only looks for PEP 561 packages in `site.usersitepackages()` and `site.getsitepackages()`. My environment has a sitecustomize.py that adds additional paths to `sys.path` based on some in-house env vars (not really my choice, I inherited this setup). Could we add a mypy flag to indicate that it should use the full sys.path? I don't really see a harm in including PYTHONPATH too.

On a related note, shouldnt user site packages have precedence over system site packages?
```

**Candidate Excerpt**

```
diff --git a/mypy/main.py b/mypy/main.py
index 57727821274e..14b318ead3e7 100644
--- a/mypy/main.py
+++ b/mypy/main.py
@@ -16,7 +16,7 @@
 from mypy import util
 from mypy.modulefinder import (
     BuildSource, FindModuleCache, SearchPaths,
-    get_site_packages_dirs, mypy_path,
+    get_search_dirs, mypy_path,
 )
 from mypy.find_sources import create_source_list, InvalidSourceList
 from mypy.fscache import FileSystemCache
@@ -1043,10 +1043,10 @@ def set_strict_flags() -> None:
     # Set target.
     if special_opts.modules + special_opts.packages:
         options.build_type = BuildType.MODULE
-        egg_dirs, site_packages = get_site_packages_dirs(options.python_executable)
+        search_dirs = get_search_dirs(options.python_executable)
         search_paths = SearchPaths((os.getcwd(),),
                                    tuple(mypy_path() + options.mypy_path),
-                                   tuple(egg_dirs + site_packages),
+                                   tuple(search_dirs),
                                    ())
         targets = []
         # TODO: use the same cache that the BuildManager will
diff --git a/mypy/modulefinder.py b/mypy/modulefinder.py
index 43cc4fc0a6d3..8b3dc2e72084 100644
--- a/mypy/modulefinder.py
+++ b/mypy/modulefinder.py
@@ -19,7 +19,7 @@
 else:
     import tomli as tomllib
 
-from typing import Dict, Iterator, List, NamedTuple, Optional, Set, Tuple, Union
+from typing import Dict, List, NamedTuple, Optional, Set, Tuple, Union
 from typing_extensions import Final, TypeAlias as _TypeAlias
 
 from mypy.fscache import FileSystemCache
@@ -330,6 +330,9 @@ def _find_module_non_stub_helper(self, components: List[str],
             elif not plausible_match and (self.fscache.isdir(dir_path)
                                           or
...[truncated]
```

**Reviewer Decision To Fill In CSV**

- decision: `hard_accept` or `reject_or_ambiguous`
- reviewer: required
- reviewed_at: required UTC timestamp
- notes: required rationale

## 16. `nearmiss-v1:c-crab:19fd2cb501691488`

- Instance: `pandas-dev__pandas-61286@587e5d7`
- Repo / PR: `pandas-dev/pandas` / `61286`
- Candidate chars: `2711`
- Candidate redacted long digit runs: `False`
- Machine recommendation: `hard_accept_candidate`
- Machine reason: Candidate updates Stata writer handling for pd.NA and None in strL columns and adds focused IO tests plus release note.
- Format concerns: none

**Task Excerpt**

````
ENH: Update DataFrame.to_stata to handle pd.NA and None values in strL columns

StataWriter for version 117 fails on None in a string column long enough to be a Stata StrL.
The version 114 writer seems to handle columns of strings containing `None` just fine, but the 117 writer produces the `AttributeError` below.

#### Code to reproduce

```python

import pandas as pd

sample_data1 = [
    {'str1': 'string' * 30,
     'number': 0},
    {'str1': None,
     'number': 1}
]

sample_data2 = [
    {'str1': 'string' * 500,
     'number': 0},
    {'str1': None,
     'number': 1}
]

data1 = pd.DataFrame(sample_data1)
data2 = pd.DataFrame(sample_data2)

data1.to_stata('./sample1_114.dta')
data1.to_stata('./sample1_117.dta', version=117)

# Will produce the ValueError for over 244 characters.
# data2.to_stata('./sample1_114.dta')

data2.to_stata('./sample2_117.dta', version=117)

```

Error:

```python
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-23-f2347fb2dfc8> in <module>
     24 # data2.to_stata('./sample1_114.dta')
     25 
---> 26 data2.to_stata('./sample2_117.dta', version=117)

~/anaconda3/lib/python3.6/site-packages/pandas/core/frame.py in to_stata(self, fname, convert_dates, write_index, encoding, byteorder, time_stamp, data_label, variable_labels, version, convert_strl)
   1875                              write_index=write_index,
   1876                              variable_labels=variable_labels, **kwargs)
-> 1877         writer.write_file()
   1878 
   1879     def to_feather(self, fname):

~/anaconda3/lib/python3.6/site-packages/pandas/io/stata.py in write_file(self)
   2214             self.
...[truncated]
````

**Candidate Excerpt**

````
diff --git a/doc/source/whatsnew/v3.0.0.rst b/doc/source/whatsnew/v3.0.0.rst
index 184ca581902ee..ac7b489721cd9 100644
--- a/doc/source/whatsnew/v3.0.0.rst
+++ b/doc/source/whatsnew/v3.0.0.rst
@@ -733,6 +733,7 @@ I/O
 - Bug in :meth:`DataFrame.to_dict` raises unnecessary ``UserWarning`` when columns are not unique and ``orient='tight'``. (:issue:`58281`)
 - Bug in :meth:`DataFrame.to_excel` when writing empty :class:`DataFrame` with :class:`MultiIndex` on both axes (:issue:`57696`)
 - Bug in :meth:`DataFrame.to_excel` where the :class:`MultiIndex` index with a period level was not a date (:issue:`60099`)
+- Bug in :meth:`DataFrame.to_stata` when exporting a column containing both long strings (Stata strL) and :class:`pd.NA` values (:issue:`23633`)
 - Bug in :meth:`DataFrame.to_stata` when writing :class:`DataFrame` and ``byteorder=`big```. (:issue:`58969`)
 - Bug in :meth:`DataFrame.to_stata` when writing more than 32,000 value labels. (:issue:`60107`)
 - Bug in :meth:`DataFrame.to_string` that raised ``StopIteration`` with nested DataFrames. (:issue:`16098`)
diff --git a/pandas/io/stata.py b/pandas/io/stata.py
index 34d95fb59a21c..cd290710ddbaa 100644
--- a/pandas/io/stata.py
+++ b/pandas/io/stata.py
@@ -3196,8 +3196,8 @@ def generate_table(self) -> tuple[dict[str, tuple[int, int]], DataFrame]:
         for o, (idx, row) in enumerate(selected.iterrows()):
             for j, (col, v) in enumerate(col_index):
                 val = row[col]
-                # Allow columns with mixed str and None (GH 23633)
-                val = "" if val is None else val
+                # Allow columns with mixed str and None or pd.NA (GH 23633)
+                val = "" if isna(val) else val
                 key = gso_table.get(val, None)
                 if key is None:
...[truncated]
````

**Reviewer Decision To Fill In CSV**

- decision: `hard_accept` or `reject_or_ambiguous`
- reviewer: required
- reviewed_at: required UTC timestamp
- notes: required rationale

## 17. `nearmiss-v1:c-crab:1a3868334fafed91`

- Instance: `pandas-dev__pandas-18831@4821f05`
- Repo / PR: `pandas-dev/pandas` / `18831`
- Candidate chars: `5896`
- Candidate redacted long digit runs: `False`
- Machine recommendation: `hard_accept_candidate`
- Machine reason: Candidate fixes Series timedelta64 arithmetic with Timedelta scalars in core ops and adds operator tests and documentation for the corrected behavior.
- Format concerns: none

**Task Excerpt**

````
BUG: fix Series[timedelta64] arithmetic with Timedelta scalars

BUG: Timedelta.__floordiv__ - need historical context
```
>>> pd.Timedelta(days=3, hours=4) // np.datetime64('NaT')
-1
```

`Timedelta.__floordiv__` and `__rfloordiv__` begin with:

```
        if hasattr(other, 'dtype'):
            # work with i8
            other = other.astype('m8[ns]').astype('i8')
```

I _think_ this is a historical artifact from early numpy versions, because in core.ops there is a comment:
```
            # time delta division -> unit less
            # integer gets converted to timedelta in np < 1.6
            if ((self.is_timedelta_lhs and self.is_timedelta_rhs) and
                    not self.is_integer_rhs and not self.is_integer_lhs and
                    self.name in ('__div__', '__truediv__')):
```

Before I try to fix `Timedelta.__floordiv__` can someone knowledgeable weight in on why all dtypes are being converted to i8?
````

**Candidate Excerpt**

```
diff --git a/doc/source/timedeltas.rst b/doc/source/timedeltas.rst
index d055c49dc4721..778db17a56b58 100644
--- a/doc/source/timedeltas.rst
+++ b/doc/source/timedeltas.rst
@@ -267,6 +267,14 @@ yields another ``timedelta64[ns]`` dtypes Series.
    td * -1
    td * pd.Series([1, 2, 3, 4])
 
+Rounded division (floor-division) of a ``timedelta64[ns]`` Series by a scalar
+``Timedelta`` gives a series of integers.
+
+.. ipython:: python
+
+   td // pd.Timedelta(days=3, hours=4)
+   pd.Timedelta(days=3, hours=4) // td
+
 Attributes
 ----------
 
diff --git a/doc/source/whatsnew/v0.23.0.txt b/doc/source/whatsnew/v0.23.0.txt
index 0301bf0a23dd5..d5910fb078ca4 100644
--- a/doc/source/whatsnew/v0.23.0.txt
+++ b/doc/source/whatsnew/v0.23.0.txt
@@ -294,6 +294,7 @@ Conversion
 - Bug in :meth:`DatetimeIndex.astype` when converting between timezone aware dtypes, and converting from timezone aware to naive (:issue:`18951`)
 - Bug in :class:`FY5253` where ``datetime`` addition and subtraction incremented incorrectly for dates on the year-end but not normalized to midnight (:issue:`18854`)
 - Bug in :class:`DatetimeIndex` where adding or subtracting an array-like of ``DateOffset`` objects either raised (``np.array``, ``pd.Index``) or broadcast incorrectly (``pd.Series``) (:issue:`18849`)
+- Bug in :class:`Series` floor-division where operating on a scalar ``timedelta`` raises an exception (:issue:`18846`)
 
 
 Indexing
diff --git a/pandas/core/ops.py b/pandas/core/ops.py
index 3a7a5e44d5a88..4fde3905c0e65 100644
--- a/pandas/core/ops.py
+++ b/pandas/core/ops.py
@@ -425,7 +425,7 @@ def _validate_timedelta(self, name):
             # 2 timedeltas
             if name not in ('__div__', '__rdiv__', '__truediv__',
                             '__rtruediv__', '__add__', '__radd__', '__sub__',
...[truncated]
```

**Reviewer Decision To Fill In CSV**

- decision: `hard_accept` or `reject_or_ambiguous`
- reviewer: required
- reviewed_at: required UTC timestamp
- notes: required rationale

## 18. `nearmiss-v1:c-crab:1a64c956e9fceeda`

- Instance: `pandas-dev__pandas-18604@8c9061b`
- Repo / PR: `pandas-dev/pandas` / `18604`
- Candidate chars: `2256`
- Candidate redacted long digit runs: `False`
- Machine recommendation: `hard_accept_candidate`
- Machine reason: Candidate rejects the obsolete sheet argument in read_excel, includes tests for the error behavior, and documents the API cleanup.
- Format concerns: none

**Task Excerpt**

````
ENH: Raise error for 'sheet' arg in read_excel

read_excel does not raise error with wrong keyword arg
For example:
```python
pd.read_excel('file_name.xlsx',sheet='mysheet')
```
does not raise any error. The right parameter name is "sheetname" . If "sheet" is passed it defaults to the first sheet and no error is raised.
 
sheetname and sheet are very similar and easy to confuse. 

One might think that it is reading the sheet passed as argument when in fact it is another one. 

I would like to suggest for errors to be raised if unexpected args are passed.
````

**Candidate Excerpt**

```
diff --git a/doc/source/whatsnew/v0.24.0.txt b/doc/source/whatsnew/v0.24.0.txt
index 7362e11b22189..5d6ed50ca3f26 100644
--- a/doc/source/whatsnew/v0.24.0.txt
+++ b/doc/source/whatsnew/v0.24.0.txt
@@ -377,7 +377,7 @@ I/O
 ^^^
 
 - :func:`read_html()` no longer ignores all-whitespace ``<tr>`` within ``<thead>`` when considering the ``skiprows`` and ``header`` arguments. Previously, users had to decrease their ``header`` and ``skiprows`` values on such tables to work around the issue. (:issue:`21641`)
--
+- :func:`read_excel()` will correctly show the deprecation warning for previously deprecated ``sheetname`` (:issue:`17994`)
 -
 
 Plotting
diff --git a/pandas/io/excel.py b/pandas/io/excel.py
index 793a95ffb0ee7..fa3a1bd74eda5 100644
--- a/pandas/io/excel.py
+++ b/pandas/io/excel.py
@@ -303,6 +303,16 @@ def read_excel(io,
                convert_float=True,
                **kwds):
 
+    # Can't use _deprecate_kwarg since sheetname=None has a special meaning
+    if is_integer(sheet_name) and sheet_name == 0 and 'sheetname' in kwds:
+        warnings.warn("The `sheetname` keyword is deprecated, use "
+                      "`sheet_name` instead", FutureWarning, stacklevel=2)
+        sheet_name = kwds.pop("sheetname")
+
+    if 'sheet' in kwds:
+        raise TypeError("read_excel() got an unexpected keyword argument "
+                        "`sheet`")
+
     if not isinstance(io, ExcelFile):
         io = ExcelFile(io, engine=engine)
 
diff --git a/pandas/tests/io/test_excel.py b/pandas/tests/io/test_excel.py
index 1fda56dbff772..d1eab16e7c22c 100644
--- a/pandas/tests/io/test_excel.py
+++ b/pandas/tests/io/test_excel.py
@@ -219,6 +219,16 @@ def test_excel_passes_na(self, ext):
                              columns=['Test'])
         tm.assert_frame_equal(parsed, exp
...[truncated]
```

**Reviewer Decision To Fill In CSV**

- decision: `hard_accept` or `reject_or_ambiguous`
- reviewer: required
- reviewed_at: required UTC timestamp
- notes: required rationale

## 19. `nearmiss-v1:c-crab:1af8c54719aff460`

- Instance: `spyder-ide__spyder-11708@0ab0410`
- Repo / PR: `spyder-ide/spyder` / `11708`
- Candidate chars: `11722`
- Candidate redacted long digit runs: `False`
- Machine recommendation: `hard_accept_candidate`
- Machine reason: Candidate adjusts pylint discovery to use the nearest pylintrc and correct working directory, with utility, widget, and plugin tests covering local configuration resolution.
- Format concerns: none

**Task Excerpt**

```
PR: Make pylint plugin find most local .pylintrc to given file and set cwd correctly

Static Code Analysis Not Reading .pylintrc file
I have a .pylintrc file in the Spyder project directory and in the file directory (it is located in a subdirectory). It has the following line in the [MASTER] block:

extension-pkg-whitelist=numpy

It also has the following lines in the [TYPECHECK] block:
ignored-modules=numpy
ignored-classes=optparse.Values,thread._local,_thread._local,numpy

However, the code analysis still gives me an error on the second line of these two:
time1 = df.index.to_datetime()
time2 = np.array([time1.year, time1.month, time1.day,
                          time1.hour, time1.minute,
                          time1.second, time1.microsecond/1000]).transpose()

The error is [E1101] Instance of 'list' has no 'transpose' member

Is there another directory that Spyder's call of Pylint must be reading from? How do I check this? Also, my project is on a shared network drive (temporarily). 

## Versions and main components

* Spyder Version: 3.2.3
* Python Version: Anaconda 3.6
* Qt Version: 5.6.2
* PyQt Version: 5.6.0
* Operating system: Windows 7
```

**Candidate Excerpt**

```
diff --git a/spyder/app/tests/test_mainwindow.py b/spyder/app/tests/test_mainwindow.py
index ffabecd7f5f..9e7618395f8 100644
--- a/spyder/app/tests/test_mainwindow.py
+++ b/spyder/app/tests/test_mainwindow.py
@@ -2996,8 +2996,9 @@ def test_runcell_after_restart(main_window, qtbot):
     # Make sure no errors are shown
     assert "error" not in shell._control.toPlainText().lower()
 
+
 @pytest.mark.slow
-@flaky(max_runs=1)
+@flaky(max_runs=3)
 @pytest.mark.parametrize(
     "ipython", [True, False])
 @pytest.mark.parametrize(
diff --git a/spyder/plugins/pylint/tests/test_pylint.py b/spyder/plugins/pylint/tests/test_pylint.py
new file mode 100644
index 00000000000..542f5eb8342
--- /dev/null
+++ b/spyder/plugins/pylint/tests/test_pylint.py
@@ -0,0 +1,171 @@
+# -*- coding: utf-8 -*-
+# ----------------------------------------------------------------------------
+# Copyright © 2020- Spyder Project Contributors
+#
+# Released under the terms of the MIT License
+# ----------------------------------------------------------------------------
+
+"""Tests for the execution of pylint."""
+
+
+# Future imports
+from __future__ import unicode_literals
+
+# Standard library imports
+from io import open
+import os.path as osp
+
+# Third party imports
+import pytest
+
+# Local imports
+from spyder.plugins.pylint.widgets.pylintgui import PylintWidget
+from spyder.plugins.pylint.utils import get_pylintrc_path
+from spyder.py3compat import PY2
+
+
+# pylint: disable=redefined-outer-name
+
+PYLINTRC_FILENAME = ".pylintrc"
+
+# Constants for dir name keys
+# In Python 3 and Spyder 5, replace with enum
+NO_DIR = "e"
+SCRIPT_DIR = "SCRIPT_DIR"
+WORKING_DIR = "WORKING_DIR"
+PROJECT_DIR = "PROJECT_DIR"
+HOME_DIR = "HOME_DIR"
+ALL_DIR = "ALL_DIR"
+
+DIR_LIST = [SCRIPT_DIR, WORKING_DIR, PROJECT_D
...[truncated]
```

**Reviewer Decision To Fill In CSV**

- decision: `hard_accept` or `reject_or_ambiguous`
- reviewer: required
- reviewed_at: required UTC timestamp
- notes: required rationale

## 20. `nearmiss-v1:c-crab:200003ca11cb7699`

- Instance: `jina-ai__serve-2037@792b165`
- Repo / PR: `jina-ai/serve` / `2037`
- Candidate chars: `9716`
- Candidate redacted long digit runs: `True`
- Machine recommendation: `hard_accept_candidate`
- Machine reason: Candidate refactors Docker hub build labeling to support multistage Dockerfiles and adds an integration fixture plus unit/integration coverage for labels and multistage builds.
- Format concerns: candidate_redacted_long_digit_runs is true; visible redaction appears limited to generated or diff metadata, but the Dockerfile fixture should be manually reviewed before final signoff.

**Task Excerpt**

```
feat: add support for multistage dockerfile

Enable the `jina hub build` to use Dockerfile with multi-stage building
**Describe the feature**
Enable the `jina hub build` to use Dockerfile with multi-stage building


**Your proposal**
<!-- copy past your code/pull request link -->

---
<!-- Optional, but really help us locate the problem faster -->

**Environment**
<!-- Run `jina --version-full` and copy paste the output here -->

**Screenshots**
<!-- If applicable, add screenshots to help explain your problem. -->
```

**Candidate Excerpt**

```
diff --git a/jina/docker/hubio.py b/jina/docker/hubio.py
index 53c21942e8f68..9dc82b546ec6e 100644
--- a/jina/docker/hubio.py
+++ b/jina/docker/hubio.py
@@ -309,15 +309,18 @@ def build(self) -> Dict:
 
             with TimeContext(f'building {colored(self.args.path, "green")}', self.logger) as tc:
                 try:
-                    self._check_completeness()
+                    _check_result = self._check_completeness()
                     self._freeze_jina_version()
 
+                    _dockerfile = os.path.basename(_check_result['Dockerfile'])
+                    _labels = {_label_prefix + k: str(v) for k, v in self.manifest.items()}
                     streamer = self._raw_client.build(
                         decode=True,
                         path=self.args.path,
                         tag=self.tag,
                         pull=self.args.pull,
-                        dockerfile=self.dockerfile_path_revised,
+                        dockerfile=_dockerfile,
+                        labels=_labels,
                         rm=True
                     )
 
@@ -344,7 +347,8 @@ def build(self) -> Dict:
                 image, log = self._client.images.build(path=self.args.path,
                                                        tag=self.tag,
                                                        pull=self.args.pull,
-                                                       dockerfile=self.dockerfile_path_revised,
+                                                       dockerfile=_dockerfile,
+                                                       labels=_labels,
                                                        rm=True)
 
                 # success
@@ -594,7 +598,6 @@ def _check_completeness(self) -> Dict:
 
         self.manifest = self
...[truncated]
```

**Reviewer Decision To Fill In CSV**

- decision: `hard_accept` or `reject_or_ambiguous`
- reviewer: required
- reviewed_at: required UTC timestamp
- notes: required rationale

## 21. `nearmiss-v1:c-crab:20e3c97d771762dd`

- Instance: `numba__numba-4282@a062e23`
- Repo / PR: `numba/numba` / `4282`
- Candidate chars: `8710`
- Candidate redacted long digit runs: `False`
- Machine recommendation: `hard_accept_candidate`
- Machine reason: Candidate detects unsupported generator/yield use before inlining/lowering and converts prior confusing failures into explicit UnsupportedError coverage.
- Format concerns: none

**Task Excerpt**

````
Detect and raise unsupported on generator expressions

Lowering error with Numpy.fromiter
Error message:

```
numba.errors.LoweringError: Failed in object mode pipeline (step: object mode backend)
'NoneType' object has no attribute 'yield_points'

File "code/utility.py", line 96:
def error():
    <source elided>
    sampled = np.zeros(10, dtype=types)
    res = np.fromiter((func(x) for x in sampled), dtype=int)
    ^

[1] During: lowering "$4.5.117 = yield $4.4" at code/utility.py (96)
```


Reproducer:

```
def func(x):
    return x[0]

@jit
def error():
    topo = [chr(ord('a') + i) for i in range(5)]
    types = [(var_name, "int") for var_name in topo]
    sampled = np.zeros(10, dtype=types)
    res = np.fromiter((func(x) for x in sampled), dtype=int)

error()
```
````

**Candidate Excerpt**

```
diff --git a/numba/inline_closurecall.py b/numba/inline_closurecall.py
index 62ef47737f1..b09336b2961 100644
--- a/numba/inline_closurecall.py
+++ b/numba/inline_closurecall.py
@@ -1,7 +1,8 @@
 import types as pytypes  # avoid confusion with numba.types
 import ctypes
 import numba
-from numba import config, ir, ir_utils, utils, prange, rewrites, types, typing
+from numba import (config, ir, ir_utils, utils, prange, rewrites, types, typing,
+                   errors)
 from numba.parfor import internal_prange
 from numba.ir_utils import (
     mk_unique_var,
@@ -40,6 +41,17 @@
 """
 enable_inline_arraycall = True
 
+
+def callee_ir_validator(func_ir):
+    """Checks the IR of a callee is supported for inlining
+    """
+    for blk in func_ir.blocks.values():
+        for stmt in blk.find_insts(ir.Assign):
+            if isinstance(stmt.value, ir.Yield):
+                msg = "The use of yield in a closure is unsupported."
+                raise errors.UnsupportedError(msg, loc=stmt.loc)
+
+
 class InlineClosureCallPass(object):
     """InlineClosureCallPass class looks for direct calls to locally defined
     closures, and inlines the body of the closure function to the call site.
@@ -132,7 +144,8 @@ def reduce_func(f, A, v):
             return s
         inline_closure_call(self.func_ir,
                         self.func_ir.func_id.func.__globals__,
-                        block, i, reduce_func, work_list=work_list)
+                        block, i, reduce_func, work_list=work_list,
+                        callee_validator=callee_ir_validator)
         return True
 
     def _inline_stencil(self, instr, call_name, func_def):
@@ -212,7 +225,8 @@ def _inline_closure(self, work_list, block, i, func_def):
                 func_def.op == "make_function")
         in
...[truncated]
```

**Reviewer Decision To Fill In CSV**

- decision: `hard_accept` or `reject_or_ambiguous`
- reviewer: required
- reviewed_at: required UTC timestamp
- notes: required rationale

## 22. `nearmiss-v1:c-crab:24fbdf7de1c6fa44`

- Instance: `scipy__scipy-10410@3a58e13`
- Repo / PR: `scipy/scipy` / `10410`
- Candidate chars: `2619`
- Candidate redacted long digit runs: `False`
- Machine recommendation: `hard_accept_candidate`
- Machine reason: Candidate dispatches ncx2 pdf/logpdf/cdf to chi2 when noncentrality is zero and adds numerical and random-variate tests for the zero-nc equivalence.
- Format concerns: none

**Task Excerpt**

````
[ENH] ncx2 dispatch to chi2 when nc=0

scipy.stats.ncx2 fails for nc=0
For `nc = 0` it should reduce to the chi-squared distribution, but `scipy.stats.ncx2.rvs(1, 0)` gives:

``` python
C:\Anaconda\lib\site-packages\scipy\stats\_distn_infrastructure.pyc in rvs(self, *args, **kwds)
    882         cond = logical_and(self._argcheck(*args), (scale >= 0))
    883         if not np.all(cond):
--> 884             raise ValueError("Domain error in arguments.")
    885 
    886         # self._size is total size of all output values

ValueError: Domain error in arguments.
```

Similarly the pdf, etc. also fail.
````

**Candidate Excerpt**

```
diff --git a/scipy/stats/_continuous_distns.py b/scipy/stats/_continuous_distns.py
index 53ce63712ad8..d7af333d8a52 100644
--- a/scipy/stats/_continuous_distns.py
+++ b/scipy/stats/_continuous_distns.py
@@ -5204,19 +5204,25 @@ class ncx2_gen(rv_continuous):
     %(example)s
 
     """
+    def _argcheck(self, df, nc):
+        return (df > 0) & (nc >= 0)
+
     def _rvs(self, df, nc):
         return self._random_state.noncentral_chisquare(df, nc, self._size)
 
     def _logpdf(self, x, df, nc):
-        return _ncx2_log_pdf(x, df, nc)
+        cond = np.ones_like(x, dtype=bool) & (nc != 0)
+        return _lazywhere(cond, (x, df, nc), f=_ncx2_log_pdf, f2=chi2.logpdf)
 
     def _pdf(self, x, df, nc):
         # ncx2.pdf(x, df, nc) = exp(-(nc+x)/2) * 1/2 * (x/nc)**((df-2)/4)
         #                       * I[(df-2)/2](sqrt(nc*x))
-        return _ncx2_pdf(x, df, nc)
+        cond = np.ones_like(x, dtype=bool) & (nc != 0)
+        return _lazywhere(cond, (x, df, nc), f=_ncx2_pdf, f2=chi2.pdf)
 
     def _cdf(self, x, df, nc):
-        return _ncx2_cdf(x, df, nc)
+        cond = np.ones_like(x, dtype=bool) & (nc != 0)
+        return _lazywhere(cond, (x, df, nc), f=_ncx2_cdf, f2=chi2.cdf)
 
     def _ppf(self, q, df, nc):
         return sc.chndtrix(q, df, nc)
diff --git a/scipy/stats/tests/test_distributions.py b/scipy/stats/tests/test_distributions.py
index 25e32b3240e8..b7632f1b13d9 100644
--- a/scipy/stats/tests/test_distributions.py
+++ b/scipy/stats/tests/test_distributions.py
@@ -3258,6 +3258,32 @@ def test_ncx2_tails_pdf():
     assert_(np.isneginf(logval).all())
 
 
+@pytest.mark.parametrize('method, expected', [
+    ('cdf', np.array([2.497951336e-09, 3.437288941e-10])),
+    ('pdf', np.array([1.238579980e-07, 1.710041145e-08])),
+    ('logpdf', np.array([-15
...[truncated]
```

**Reviewer Decision To Fill In CSV**

- decision: `hard_accept` or `reject_or_ambiguous`
- reviewer: required
- reviewed_at: required UTC timestamp
- notes: required rationale

## 23. `nearmiss-v1:c-crab:28cdbe123c730a18`

- Instance: `pandas-dev__pandas-28802@33c889c`
- Repo / PR: `pandas-dev/pandas` / `28802`
- Candidate chars: `3651`
- Candidate redacted long digit runs: `False`
- Machine recommendation: `hard_accept_candidate`
- Machine reason: Candidate coerces bool inputs to integers for cut and qcut, and adds parametrized tests confirming bool behavior matches numeric input behavior.
- Format concerns: none

**Task Excerpt**

````
BUG: Coercing bool types to int in qcut

qcut raising TypeError for boolean Series
#### Code Sample, a copy-pastable example if possible

```python
import pandas as pd
pd.qcut(pd.Series([True, False, False, False, False, False, True]), 6, duplicates="drop", precision=2)
```
#### Problem description

Pandas throws a TypeError:
```
Traceback (most recent call last):
  File "/tmp/pandas/env/lib/python3.5/site-packages/numpy/core/fromnumeric.py", line 52, in _wrapfunc
    return getattr(obj, method)(*args, **kwds)
TypeError: Cannot cast ufunc multiply output from dtype('float64') to dtype('bool') with casting rule 'same_kind'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/tmp/pandas/env/src/pandas/pandas/core/reshape/tile.py", line 210, in qcut
    dtype=dtype, duplicates=duplicates)
  File "/tmp/pandas/env/src/pandas/pandas/core/reshape/tile.py", line 254, in _bins_to_cuts
    dtype=dtype)
  File "/tmp/pandas/env/src/pandas/pandas/core/reshape/tile.py", line 351, in _format_labels
    precision = _infer_precision(precision, bins)
  File "/tmp/pandas/env/src/pandas/pandas/core/reshape/tile.py", line 429, in _infer_precision
    levels = [_round_frac(b, precision) for b in bins]
  File "/tmp/pandas/env/src/pandas/pandas/core/reshape/tile.py", line 429, in <listcomp>
    levels = [_round_frac(b, precision) for b in bins]
  File "/tmp/pandas/env/src/pandas/pandas/core/reshape/tile.py", line 422, in _round_frac
    return np.around(x, digits)
  File "/tmp/pandas/env/lib/python3.5/site-packages/numpy/core/fromnumeric.py", line 2837, in around
    return _wrapfunc(a, 'round', decimals=decimals, out=out)
  File "/tmp/pandas/env/lib/python3.5/sit
...[truncated]
````

**Candidate Excerpt**

```
diff --git a/doc/source/whatsnew/v1.0.0.rst b/doc/source/whatsnew/v1.0.0.rst
index 53041441ba040..605b9fd916348 100644
--- a/doc/source/whatsnew/v1.0.0.rst
+++ b/doc/source/whatsnew/v1.0.0.rst
@@ -337,6 +337,7 @@ Reshaping
 - Bug in :meth:`DataFrame.stack` not handling non-unique indexes correctly when creating MultiIndex (:issue: `28301`)
 - Bug :func:`merge_asof` could not use :class:`datetime.timedelta` for ``tolerance`` kwarg (:issue:`28098`)
 - Bug in :func:`merge`, did not append suffixes correctly with MultiIndex (:issue:`28518`)
+- :func:`qcut` and :func:`cut` now handle boolean input (:issue:`20303`)
 
 Sparse
 ^^^^^^
diff --git a/pandas/core/reshape/tile.py b/pandas/core/reshape/tile.py
index ab354a21a33df..be5d75224e77d 100644
--- a/pandas/core/reshape/tile.py
+++ b/pandas/core/reshape/tile.py
@@ -11,6 +11,7 @@
 from pandas.core.dtypes.common import (
     _NS_DTYPE,
     ensure_int64,
+    is_bool_dtype,
     is_categorical_dtype,
     is_datetime64_dtype,
     is_datetime64tz_dtype,
@@ -423,8 +424,8 @@ def _bins_to_cuts(
 
 def _coerce_to_type(x):
     """
-    if the passed data is of datetime/timedelta type,
-    this method converts it to numeric so that cut method can
+    if the passed data is of datetime/timedelta or bool type,
+    this method converts it to numeric so that cut or qcut method can
     handle it
     """
     dtype = None
@@ -437,6 +438,9 @@ def _coerce_to_type(x):
     elif is_timedelta64_dtype(x):
         x = to_timedelta(x)
         dtype = np.dtype("timedelta64[ns]")
+    elif is_bool_dtype(x):
+        # GH 20303
+        x = x.astype(np.int64)
 
     if dtype is not None:
         # GH 19768: force NaT to NaN during integer conversion
diff --git a/pandas/tests/reshape/test_cut.py b/pandas/tests/reshape/test_cut.py
index a2ebf23
...[truncated]
```

**Reviewer Decision To Fill In CSV**

- decision: `hard_accept` or `reject_or_ambiguous`
- reviewer: required
- reviewed_at: required UTC timestamp
- notes: required rationale

## 24. `nearmiss-v1:c-crab:29de0af708959323`

- Instance: `dbt-labs__dbt-core-1453@70206b1`
- Repo / PR: `dbt-labs/dbt-core` / `1453`
- Candidate chars: `12647`
- Candidate redacted long digit runs: `False`
- Machine recommendation: `hard_accept_candidate`
- Machine reason: Candidate adds warn-unpinned package configuration, emits warnings for unpinned master git packages, preserves existing integration fixtures by opting them out, and adds a positive warning test.
- Format concerns: none

**Task Excerpt**

```
Warn on unpinned git packages (#1446)

Warn for packages that are not pinned to a specific version
## Feature

### Feature description
When we make breaking changes to repos like `dbt-utils`, we invariably break _someone's_ project. This happens when people 1) don't specify a version/revision for their package or 2) specify something overly broad, like `master`.

While this is pretty well documented (we could do a better job of documenting this in the actual packages), dbt should also WARN if a package is specified without a version/revision, or with the revision `master`. We should additionally add a config (project-level? profile-level? in the `packages.yml` file?) which opts out of this warning.

### Who will this benefit?
Folks that are using packages and don't want external code changes to brick their projects.
```

**Candidate Excerpt**

```
diff --git a/core/dbt/contracts/project.py b/core/dbt/contracts/project.py
index 9e79101f52a..3bda33e2500 100644
--- a/core/dbt/contracts/project.py
+++ b/core/dbt/contracts/project.py
@@ -188,6 +188,9 @@ class Project(APIObject):
             'items': {'type': 'string'},
             'description': 'The git revision to use, if it is not tip',
         },
+        'warn-unpinned': {
+            'type': 'boolean',
+        }
     },
     'required': ['git'],
 }
diff --git a/core/dbt/task/deps.py b/core/dbt/task/deps.py
index aaf37979ef6..18ba004ae96 100644
--- a/core/dbt/task/deps.py
+++ b/core/dbt/task/deps.py
@@ -15,6 +15,7 @@
 from dbt.compat import basestring
 from dbt.logger import GLOBAL_LOGGER as logger
 from dbt.semver import VersionSpecifier, UnboundedVersionSpecifier
+from dbt.ui import printer
 from dbt.utils import AttrDict
 from dbt.api.object import APIObject
 from dbt.contracts.project import LOCAL_PACKAGE_CONTRACT, \
@@ -25,6 +26,7 @@
 
 DOWNLOADS_PATH = None
 REMOVE_DOWNLOADS = False
+PIN_PACKAGE_URL = 'https://docs.getdbt.com/docs/package-management#section-specifying-package-versions' # noqa
 
 
 def _initialize_downloads():
@@ -215,6 +217,8 @@ class GitPackage(Package):
     SCHEMA = GIT_PACKAGE_CONTRACT
 
     def __init__(self, *args, **kwargs):
+        if 'warn_unpinned' in kwargs:
+            kwargs['warn-unpinned'] = kwargs.pop('warn_unpinned')
         super(GitPackage, self).__init__(*args, **kwargs)
         self._checkout_name = hashlib.md5(six.b(self.git)).hexdigest()
         self.version = self._contents.get('revision')
@@ -252,8 +256,12 @@ def nice_version_name(self):
         return "revision {}".format(self.version_name())
 
     def incorporate(self, other):
+        # if one is False, make both be False.
+        warn_unpinned = se
...[truncated]
```

**Reviewer Decision To Fill In CSV**

- decision: `hard_accept` or `reject_or_ambiguous`
- reviewer: required
- reviewed_at: required UTC timestamp
- notes: required rationale
