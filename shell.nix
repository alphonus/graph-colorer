{ pkgs ? import <nixpkgs> {} }:
(pkgs.buildFHSUserEnv {
  name = "model";
  targetPkgs = pkgs: (with pkgs; [
    python310
    python310Packages.pip
    python310Packages.virtualenv
    cudaPackages_11_7.cudatoolkit
    #gcc_multi
    #libgccjit
    jetbrains.pycharm-professional
  ]);
  runScript = pkgs.writeScript "init.sh" ''
  source /home/martin/.venvs/modelvenv10/bin/activate
  exec bash
  code
  '';
}).env

