from src import ModelOptions, main

options = ModelOptions().parse()
options.mode = 4
main(options)
