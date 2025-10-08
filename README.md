# High-Performance BitCrack Fork

The original README can be found [here](https://github.com/brichard19/BitCrack).

## Changes Made

OpenCL has been completely removed.
CUDA builds by default without specifying the `BUILD_CUDA` environment variable.

### Memory Usage Optimization

Completely redesigned initialization.
In the original version, keys were created on the host and copied to GPU, then points were formed from these keys.
After point formation, keys were cleared from GPU, leaving more than 25% of memory unused.

Now a single key is formed on the host.
GPU calculates the required key offset for its thread on-the-fly and forms its own point.
Result - ability to use almost all memory.

### Performance Optimization

Used coalesced memory access patterns, which significantly improved performance without breaking the original logic.

For example: A modest RTX 4060 on the original build achieved `750-770 MKey/s`.
- In this build - `950+ MKey/s` with `-b 640 -t 128 -p 1024` (`83,886,080 starting points`, memory fully utilized at 100%).
- Found even more optimal parameters for RTX 4060... something like `-b 240 -t 256 -p 1024` (`62,914,560 starting points`, memory `5878 / 7805MB`).

Actually, measurements showed that the computational potential is much higher.
Specifically, RTX 4060 is capable of "adding" up to 1500 MKey/s.
The problem is in the memory bus bandwidth.

RTX 4090: `-b 1950 -t 128 -p 1024` - `3520+ MKey/s` (`255,590,400 starting point`, memory `23797 / 24080MB`).

### Cosmetic Improvements

- Added WIF format to output.
- Added Telegram notification option to argument list.
- Fixed Bloom filter bug. For those who didn't understand - you can now specify more than 16 targets.
- If there's an error in the targets file, the system won't stop. It will notify how many lines were skipped and start processing without invalid ones.
- Dependency is configured for CUDA 13.0, but can be downgraded to 12.x. Tested on 12.8.
- Updated compilation flags.
- Lots of minor improvements, fixes, and optimizations.

## Building

Builds under Linux (tested on Ubuntu) and Windows.
Compared to the original version - nothing new: set your `sm cap` and build, without the `BUILD_CUDA` flag.
By default, `sm cap 89` is set (RTX 40xx).

## Disclaimer

_This software is provided "as is", without any warranties._
_It is intended exclusively for research purposes: studying performance optimization on the CUDA platform._

_**Hardware Risk Warning:** This code creates extreme load on graphics processors, which may lead to overheating and failure if used improperly._
_You use it at your own risk._
_The developer is not responsible for any damage caused to your hardware._

_**Important Legal Notice:** The developer does not encourage or support the use of this code for any illegal activities, including but not limited to attempts at unauthorized access to wallets or third-party data._
_Any such use violates the laws of most countries._
_All responsibility for using the program lies with the end user._