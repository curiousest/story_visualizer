import getopt
import sys

from image_retriever import set_image_for_text

CORRECT_COMMAND = "set_image_for_text.py -i <imageurl> -t <text> -p <partofspeech>"


def main(argv):
    try:
        opts, args = getopt.getopt(
            argv, "hi:t:p:", ["imageurl=", "text=", "partofspeech="]
        )
    except getopt.GetoptError as e:
        print(str(e))
        print(CORRECT_COMMAND)
        sys.exit(2)
    image_url = None
    text = None
    part_of_speech = None
    for opt, arg in opts:
        if opt == "-h":
            print(CORRECT_COMMAND)
            sys.exit()
        elif opt in ("-i", "--imageurl"):
            image_url = arg
        elif opt in ("-t", "--text"):
            text = arg
        elif opt in ("-p", "--partofspeech"):
            part_of_speech = arg
    if not (image_url and text and part_of_speech):
        print(CORRECT_COMMAND)
        sys.exit()
    result = set_image_for_text(url=image_url, word=text, part_of_speech=part_of_speech)
    print(f"Image set for {text} {part_of_speech}, in {result.image_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
