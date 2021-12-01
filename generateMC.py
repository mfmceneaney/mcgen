from mclib import Generator
from numpy.random import randint

# Utility Imports
import argparse, os

def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description='Run Worm Algorithm Monte Carlo on Hard Core Boson Lattice')
    parser.add_argument('--N', type=int, default=2,
                        help='# of lattice sites along a spatial axis (default: 2)')
    parser.add_argument('--d', type=int, default=3,
                        help='# of spatial dimensions (default: 3)')
    parser.add_argument('--b', type=float, default=12.0,
                        help='beta parameter (default: 12.0)')
    parser.add_argument('--mu', type=float, default=1.4,
                        help='mu parameter (default: 1.4)')
    parser.add_argument('--e', type=float, default=0.01,
                        help='epsilon parameter (default: 0.01)')
    parser.add_argument('--t', type=float, default=1.0,
                        help='t parameter (default: 1.0)')
    parser.add_argument('--seed', type=int, default=-1,
                        help='Number of spatial dimensions (default: random seed)')
    parser.add_argument('--n', type=int, default=10000,
                        help='Max # of configurations to generate (default: 10000)')
    parser.add_argument('--maxEntries', type=int, default=10000,
                        help='Max entries per file (default: 10000)')

    # Output directory option
    parser.add_argument('--log', type=str, default='mcgen/',
                        help='Log directory for histograms (default: mcgen/)')

    # Input directory option #If you want to extend this for reading in data
    parser.add_argument('--path', type=str, default='mcgen/',
                        help='Data load path (default: mcgen/)')

    # Input dataset directory prefix option
    parser.add_argument('--name', type=str, default='mcgen.txt',
                        help='Output text file basename (default: mcgen.txt)')

    args = parser.parse_args()

    # Setup log directory
    try: os.mkdir(args.log)
    except FileExistsError: print('Directory:',args.log,'already exists!')

    # Initialize generator
    g = Generator(args.e,
        t=args.t,
        beta=args.b,
        mu=args.mu,
        N=[args.N for k in range(args.d)], #NOTE: I think we just have to deal with cubic lattices anyway.
        seed=args.seed if args.seed==-1 else randint(0,1000),
        fileName=args.log+args.name,
        maxEntries=args.maxEntries)

    # Generate data
    g.loop(args.n,obs=["N","E","ET","W","W2","SHops","THops"])  #TODO: Could figure out how to get obs from argparse object, not really sure how

if __name__ == '__main__':

    main()
