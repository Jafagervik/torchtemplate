from argparse import ArgumentParser


def argsparser():
    parser = ArgumentParser(prog="SugmaNet", description="no", epilog="byebye")
    parser.add_argument("model")
    parser.add_argument("batch_size")
    parser.add_argument("lr")
    parser.add_argument("epochs")


if __name__ == "__main__":
    args = argsparser()
