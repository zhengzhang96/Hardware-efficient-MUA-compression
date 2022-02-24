`include "params.v"
/*
Calculate the length of total codeword length using different encoders and select the shortest one.

Bit shifts are mannal set according to the codeword length
*/
module selector_3(
//input CLK,
input RST,
input finish_in,

input [`FREQ_BIT-1:0] freq1,
input [`FREQ_BIT-1:0] freq2,
input [`FREQ_BIT-1:0] freq3,
input [`FREQ_BIT-1:0] freq4,
input [`FREQ_BIT-1:0] freq5,

//input [`SPIKE_RATE_BIT-1:0] max_rate,
//input [0:`ENCODER_LENGTH*`SPIKE_RATE_CLIP-1] len,

//output reg [`ENCODER_NUM_BIT-1:0]encoder_addr,
output [`ENCODER_NUM_BIT-1:0]encoder_sel
);


wire [`TOTAL_LEN_BIT-1:0] total_len1,total_len2,total_len3,total_len4,total_len5;
wire [`TOTAL_LEN_BIT-1:0]freq1_,freq2_,freq3_,freq4_,freq5_;
wire encoder_sel_;


assign freq1_ = freq1;
assign freq2_ = freq2;
assign freq3_ = freq3;
assign freq4_ = freq4;
assign freq5_ = freq5;

assign total_len1 = freq1_+(freq2_<<1)+freq3_+(freq3_<<1)+(freq4_<<2)+(freq5_<<2);
assign total_len2 = (freq1_<<1)+(freq2_<<1)+(freq3_<<1)+freq4_+(freq4_<<1)+freq5_+(freq5_<<1);
assign total_len3 = freq1_+freq2_+(freq2_<<1)+freq3_+(freq3_<<1)+freq4_+(freq4_<<1)+freq5_+(freq5_<<1);	



assign encoder_sel = (total_len1<=total_len2) && (total_len1<=total_len3)?0:
			     (total_len2<=total_len1) && (total_len2<=total_len3)?1:2;


/*always@(posedge finish_in or negedge RST) begin
	if(!RST)
		finish_out <= 0;
	else begin
 		finish_out <= 1;
 		encoder_sel <= encoder_sel_;
	end
end*/

		
	
endmodule
