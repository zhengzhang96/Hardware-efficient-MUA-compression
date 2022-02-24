`include "params.v"
/*This is a just binning implementation*/
module whole_system(clk_,RST,detect,calibration,finish_out,codeword,length);
input clk_,RST,detect,calibration;
output [`MAX_CODEWORD_LENGTH-1:0] codeword;
output [`LENGTH_WIDTH-1:0] length;
output finish_out;

wire clk, CLK;
wire[`SPIKE_RATE_BIT-1:0] spike_number,spike_number_in,spike_number_out,spike_number_out_,max_rate,max_rate_,rate_out;
wire [`FREQ_BIT-1:0] freq_count1,freq_count2,freq_count3,freq_count4,freq_count5,freq_1,freq_2,freq_3,freq_4,freq_5;
wire bin_finish;
//wire[`BIN_PERIOD_WIDTH-1:0] bin_period;



//wire [`SPIKE_RATE_BIT-1:0]spike_rate1,spike_rate2,spike_rate3,spike_rate4,spike_rate5;//spike_rate6,spike_rate7,spike_rate8,spike_rate9,spike_rate10,spike_rate11,spike_rate12,spike_rate13,spike_rate14,spike_rate15;
//wire [`FREQ_BIT-1:0] freq_count1,freq_count2,freq_count3,freq_count4,freq_count5,freq_count6,freq_count7,freq_count8,freq_count9,freq_count10,freq_count11,freq_count12,freq_count13,freq_count14,freq_count15;

wire [`SPIKE_RATE_BIT*2+`ENCODER_NUM_BIT-1:0] ram_in, ram_out;
wire [`ENCODER_NUM_BIT-1:0] encoder_sel,encoder_sel_;

//wire [`SPIKE_RATE_BIT-1:0] max_rate;
wire [`CH_BIT-1:0] addr,channel_count;
reg [`CH_BIT-1:0] c_channel;





/*assign bin_period = 50;
assign waddr = channel_count == 0? `CH_NUM - 1 : channel_count - 1;

binner BIN(CLK,RST,detect,spike_number_in, channel_count, spike_number,spike_number_out,bin_finish);

dualRam DR(spike_number, we,channel_count, ~clk, channel_count, clk, spike_number_in);

clock_divier CD(clk_,RST,clk,CLK);

encoder_2 E2(bin_finish, 1'b1, spike_number_out, codeword, length);*/

//multichannel_encoder ME(clk_,RST,detect,we,codeword,length)


//assign addr = calibration?c_channel:channel_count;
reg cali_finish;

wire triger;

assign triger = ~CLK;
assign addr = c_channel;

always@(posedge triger or negedge RST) begin
	if(~RST) begin
		c_channel <= 0;
	end
	else if(cali_finish == 0) begin
		c_channel <= c_channel + 1;
	end else 
		if(c_channel == `CH_NUM - 1 || c_channel == `CH_NUM )
			c_channel <= 0;
		else c_channel <= c_channel + 1;
end

always@(negedge finish_out or posedge calibration) begin
	if(calibration)	cali_finish <= 0;
	else if(!finish_out && c_channel == `CH_NUM ) cali_finish <= 1;
	else cali_finish <= cali_finish;
end
	 	

 
binner_f BIN(CLK,RST,detect,cali_finish,spike_number_in,c_channel,spike_number,spike_number_out,spike_number_out_, bin_finish);


assign ram_in = {spike_number, max_rate_, encoder_sel_};
assign spike_number_in = ram_out[`SPIKE_RATE_BIT*2+`ENCODER_NUM_BIT-1:`SPIKE_RATE_BIT+`ENCODER_NUM_BIT];
assign max_rate_ = ram_out[`SPIKE_RATE_BIT+`ENCODER_NUM_BIT-1:`ENCODER_NUM_BIT];
assign encoder_sel_ = ram_out[`ENCODER_NUM_BIT-1:0];

dualRam DR(ram_in, 1,addr, ~clk, addr, clk, ram_out);

clock_divier CD(clk_,RST,clk,CLK);

mapper MP(spike_number_out,max_rate_,rate_out);

encoder_3 E3(bin_finish, encoder_sel_, rate_out, codeword, length);

endmodule