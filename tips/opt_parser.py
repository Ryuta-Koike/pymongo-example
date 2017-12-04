import argparse

def display(MODE, FILE):
    print('MODE:', MODE, 'FILE:', FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("MODE", help="Specify mode",type=str)
    parser.add_argument("-f", "--FILE", help="Specify file. default=this",type=str,
                        nargs='?',default='this',const='this')
    args = parser.parse_args()
    display(args.MODE, args.FILE)
