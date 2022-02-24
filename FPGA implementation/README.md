This is the multichannel implementation of the compression algorithm

Please open compression_multichannel.mpf with Modelsim. If file not found, recreate the project and import everything.

Module archtecure:

|-multichannel_encoder ------------------	A stand-alone implementation	
|-tb_whole_system------------------------	The test banch for this design
	|--1_binner_final---------------	Multi-channel binner counts the number of detections in a bin_period
	|--2_hist------------------------	Counts the histogram of different spike rate
	|--3_bruforse_sorter-----------	Estimate the Sorted the histogram frequency
	|--4_1_mapper-----------------	Map the spike rate according to the sorted result
	|--4_selector_3-----------------  Calculate the total codeword length and obtain the encoder gives least number of bits
	|--5_encoder_3-----------------	Encoders the mapped spike rate
	|--RAM-------------------------	Stores the current channel's spike rate, max frequency spike rate and selected encoder 
	|--clock_divider----------------	Generate the memory clock and processing clock
	|--param-----------------------	All parameters in this syste,
	|--mem------------------------	Memory initial values (0s)
	|--binned_MUA_1_aligned----	Data used to be compression -> input order |CH1|CH2|CH3|...|CHN|CH1|CH2|....
	|--binned_MUA_1_aligned_---	Data used for calibration -> input order |CH1|CH1|CH1|...|CH2|CH2|CH2|....|CH...|....|CHN|CHN|CHN|....|

Procedure:

Calibration phase:
CH1: binner -> hist -> sorter -> selector -> spike rate with highest frequency from hist and selected encoder number from selector will be stored in RAM address 1
CH2:...
CH3:...
...
CHN:....

Compression phase:
CH1: 	binner reads the current spike rate from RAM
	Increase it by one if a spike is detection
	Store back to RAM and read next line
	If bin period reached
		mapper map the binned spike rate to the correct number
		encoder encode it and provide the codeword length
CH2:....
....


