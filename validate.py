from src import ModelOptions, main

options = ModelOptions().parse()
options.mode = 3
main(options)
