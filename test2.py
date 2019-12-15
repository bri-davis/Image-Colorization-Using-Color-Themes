from src import ModelOptions, main

options = ModelOptions().parse()
options.mode = 5
main(options)
