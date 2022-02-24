`include "params.v"
/*
A lookup table for sorting the frequency of spike rate.
*/
module brute_sorter(
input RST,
input finish_in,
input [`FREQ_BIT-1:0] freq1,
input [`FREQ_BIT-1:0] freq2,
input [`FREQ_BIT-1:0] freq3,
input [`FREQ_BIT-1:0] freq4,
input [`FREQ_BIT-1:0] freq5,


input [`SPIKE_RATE_BIT-1:0] max_rate,
output [`FREQ_BIT-1:0] freq_1,
output [`FREQ_BIT-1:0] freq_2,
output [`FREQ_BIT-1:0] freq_3,
output [`FREQ_BIT-1:0] freq_4,
output [`FREQ_BIT-1:0] freq_5


);
//reg [`SPIKE_RATE_BIT-1:0] max_rate;

/*always@(posedge finish_in or negedge RST)
	if(!RST)
		finish_out <= 0;
	else begin
		max_rate <= max_rate_;
		finish_out <= 1;
	end*/

assign freq_1 = max_rate == 0? freq1:max_rate == 1? freq2:max_rate == 2? freq4:max_rate == 3? freq5:freq5;
assign freq_2 = max_rate == 0? freq2:max_rate == 1? freq1:max_rate == 2? freq2:max_rate == 3? freq4:freq4;
assign freq_3 = max_rate == 0? freq3:max_rate == 1? freq3:max_rate == 2? freq1:max_rate == 3? freq2:freq3;
assign freq_4 = max_rate == 0? freq4:max_rate == 1? freq5:max_rate == 2? freq3:max_rate == 3? freq1:freq2;
assign freq_5 = max_rate == 0? freq5:max_rate == 1? freq5:max_rate == 2? freq5:max_rate == 3? freq3:freq1;

/*assign spike_rate1 = max_rate == 0? 0:max_rate == 1? 1:max_rate == 2?3:max_rate == 3? 4:4;
assign spike_rate2 = max_rate == 0? 1:max_rate == 1? 0:max_rate == 2?1:max_rate == 3? 3:3;
assign spike_rate3 = max_rate == 0? 2:max_rate == 1? 2:max_rate == 2?0:max_rate == 3? 1:2;
assign spike_rate4 = max_rate == 0? 3:max_rate == 1? 3:max_rate == 2?2:max_rate == 3? 0:1;
assign spike_rate5 = max_rate == 0? 4:max_rate == 1? 4:max_rate == 2?4:max_rate == 3? 2:0;*/
endmodule