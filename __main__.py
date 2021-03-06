"""Steganographer Tool."""
import os
import imageio
#import click
from steganographer import Steganographer
import sys


def main(embed, decode, image_file, message_file):
    """Run Steganographer tool."""
    method = ""
    if embed:
        print("METHOD embed")
        method = "embed"
    elif decode:
        method = "decode"
        print("METHOD decode")

    input_dir = "input"

    # Load Image
    if not os.path.exists(input_dir):
        print("Error: Input Directory does not exist")
        exit(1)

    if not os.path.isfile(input_dir + '/' + image_file):
        print("Error: Image file "
              + image_file + " does not exist in input folder.")
        exit(1)
    image_in = imageio.imread(input_dir + '/' + image_file)

    # Load Message
    message_in = ""
    if method == "embed":
        if not os.path.isfile(input_dir + '/' + message_file):
            print("Error: Message file "
                  + message_file + " does not exit in input folder.")

        file = open(input_dir + '/' + message_file, 'r')
        message_in = file.read()

    # Run Steganography Tool
    print(method)
    stego = Steganographer(method, image_in,
                           image_file, message_in, message_file)
    stego.run()


if __name__ == '__main__':
    
    # pylint: disable=no-value-for-parameter
    if sys.argv[1]=='embed':
        embed=True
        decode=False
        image_file = sys.argv[2]
        message_file = sys.argv[3]
        main(embed,decode,image_file,message_file)
    elif sys.argv[1]=='decode':
        decode=True
        embed=False
        image_file=sys.argv[2]
        main(embed,decode,image_file,None)
    else:
        print("enter a valid method")
        exit()

    

    #main(embed,decode,image_file,message_file)
