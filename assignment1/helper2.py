Jupyter Notebook requires JavaScript.  
Please enable it to proceed.

[ ![Jupyter Notebook](helper_files/logo.png)
](http://localhost:8888/tree?token=574148da1368cd07381b5f8fcba43fd08b00d143070293ed
"dashboard")

helper.py

12 minutes ago Logout

Python

__ Menu

  * 

  * File
    * New
    * Save
    * Rename
    * Download
  * Edit
    * Find
    * Find & Replace
    *     * Key Map
    * Default __
    * Sublime Text __
    * Vim __
    * emacs __
  * View
    * Toggle Header
    * Toggle Line Numbers
  * Language
    * APL
    * PGP
    * ASN.1
    * Asterisk
    * Brainfuck
    * C
    * C++
    * Cobol
    * C#
    * Clojure
    * ClojureScript
    * Closure Stylesheets (GSS)
    * CMake
    * CoffeeScript
    * Common Lisp
    * Cypher
    * Cython
    * Crystal
    * CSS
    * CQL
    * D
    * Dart
    * diff
    * Django
    * Dockerfile
    * DTD
    * Dylan
    * EBNF
    * ECL
    * edn
    * Eiffel
    * Elm
    * Embedded Javascript
    * Embedded Ruby
    * Erlang
    * Esper
    * Factor
    * FCL
    * Forth
    * Fortran
    * F#
    * Gas
    * Gherkin
    * GitHub Flavored Markdown
    * Go
    * Groovy
    * HAML
    * Haskell
    * Haskell (Literate)
    * Haxe
    * HXML
    * ASP.NET
    * HTML
    * HTTP
    * IDL
    * Pug
    * Java
    * Java Server Pages
    * JavaScript
    * JSON
    * JSON-LD
    * JSX
    * Jinja2
    * Julia
    * Kotlin
    * LESS
    * LiveScript
    * Lua
    * Markdown
    * mIRC
    * MariaDB SQL
    * Mathematica
    * Modelica
    * MUMPS
    * MS SQL
    * mbox
    * MySQL
    * Nginx
    * NSIS
    * NTriples
    * Objective-C
    * OCaml
    * Octave
    * Oz
    * Pascal
    * PEG.js
    * Perl
    * PHP
    * Pig
    * Plain Text
    * PLSQL
    * PowerShell
    * Properties files
    * ProtoBuf
    * Python
    * Puppet
    * Q
    * R
    * reStructuredText
    * RPM Changes
    * RPM Spec
    * Ruby
    * Rust
    * SAS
    * Sass
    * Scala
    * Scheme
    * SCSS
    * Shell
    * Sieve
    * Slim
    * Smalltalk
    * Smarty
    * Solr
    * SML
    * Soy
    * SPARQL
    * Spreadsheet
    * SQL
    * SQLite
    * Squirrel
    * Stylus
    * Swift
    * sTeX
    * LaTeX
    * SystemVerilog
    * Tcl
    * Textile
    * TiddlyWiki 
    * Tiki wiki
    * TOML
    * Tornado
    * troff
    * TTCN
    * TTCN_CFG
    * Turtle
    * TypeScript
    * TypeScript-JSX
    * Twig
    * Web IDL
    * VB.NET
    * VBScript
    * Velocity
    * Verilog
    * VHDL
    * Vue.js Component
    * XML
    * XQuery
    * Yacas
    * YAML
    * Z80
    * mscgen
    * xu
    * msgenny

    
    
    x



1

    
    
    import numpy as np 

2

    
    
    import cv2

3

    
    
    from scipy.linalg import inv, cholesky,svd, null_space

4

    
    
    import matplotlib.pyplot as plt

5

    
    
    ​

6

    
    
    def construct2rows(list_):

7

    
    
        x1,y1,x2,y2 = list_

8

    
    
        row1 = np.array([x1,y1,1,0,0,0,-1*x1*x2,-1*y1*x2])

9

    
    
        row2 = np.array([0,0,0,x1,y1,1,-1*x1*y2,-1*y1*y2])

10

    
    
        return np.stack([row1,row2])

11

    
    
    def point_corr_H(point_list):

12

    
    
        A = [construct2rows(list_) for list_ in point_list]

13

    
    
        A = np.concatenate(A).astype('float64')

14

    
    
        b = point_list[:,2:].flatten()

15

    
    
        print('The A matrix for H :',A)

16

    
    
        print('The b vector for H',b)

17

    
    
        return np.linalg.solve(A,b)

18

    
    
    ​

19

    
    
    def construct_grid(width, height, homogenous):

20

    
    
        coords = np.indices((width, height)).reshape(2, -1)

21

    
    
        return np.vstack((coords, np.ones(coords.shape[1]))).astype(np.int) if homogenous else coords

22

    
    
    ​

23

    
    
    def get_H(point_list):

24

    
    
        H = point_corr_H(point_list)

25

    
    
        print("Homography estimate: ",H)

26

    
    
        H = H.tolist()

27

    
    
        H.append(1)

28

    
    
        H = np.array(H,dtype='float64').reshape(3,3)

29

    
    
        return H

30

    
    
    ​

31

    
    
    def get_arr_from_H(img,width,height,H):

32

    
    
        grid = construct_grid(width,height,True).astype(int)

33

    
    
        # print(H_perp.shape)

34

    
    
        aff_grid = np.matmul(H,grid)

35

    
    
        print(aff_grid.shape)

36

    
    
        x_aff = np.round(np.divide(aff_grid[0],aff_grid[2])).astype(int)

37

    
    
        y_aff = np.round(np.divide(aff_grid[1],aff_grid[2])).astype(int)

38

    
    
        print(x_aff)

39

    
    
        x_aff -= min(x_aff)

40

    
    
        y_aff -= min(y_aff)

41

    
    
        inds = np.where((x_aff>= 0) & (y_aff>=0) & (x_aff<width) & (y_aff<height))

42

    
    
    ​

