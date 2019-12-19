{ pkgs ? import <nixpkgs> { }, ... }:

pkgs.stdenvNoCC.mkDerivation rec {
  name = "pycl-env";
  env = pkgs.buildEnv { name = name; paths = buildInputs; };

  buildInputs = let
    local-python = pkgs.python3.withPackages (python-packages: with python-packages; [
      numpy
      sympy
      pyopencl setuptools
      matplotlib
    ]);

  in [
    local-python
    pkgs.opencl-info
    pkgs.universal-ctags
  ];

  shellHook = ''
    export NIX_SHELL_NAME="${name}"
    export PYOPENCL_COMPILER_OUTPUT=1
    export PYTHONPATH="$PWD:$PYTHONPATH"
  '';
}
