import argparse
from src.models.delete_baseline import predict_baseline


def main(args):

    save_path1 = args.save_path1
    save_path2 = args.save_path2
    save_path3 = args.save_path3
    
    #create output file
    predict_baseline(save_path1, save_path2, save_path3)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path1", help="save path to reference.txt", default="../data/interim/reference.txt")
    parser.add_argument("--save_path2", help="save path to bad_words.txt", default='../data/interim/bad_words.txt')
    parser.add_argument("--save_path3", help="save path to baseline.txt", default='../data/interim/baseline.txt')

    args = parser.parse_args()


    main(args)