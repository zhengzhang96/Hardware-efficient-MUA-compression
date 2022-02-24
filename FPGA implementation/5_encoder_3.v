`include "params.v"
/*
Encode the mapped output to the codeword and provide the codeword length for serilisation.
*/
module encoder_3(
	input bin_finish,
	input [`ENCODER_NUM_BIT-1:0] encoder_sel,
	input [`SPIKE_RATE_BIT-1:0]spike_number,
	output reg [`MAX_CODEWORD_LENGTH-1:0] codeword,
	output reg [`LENGTH_WIDTH-1:0] length
);
always@(*) begin
	
	if (bin_finish)
		case (encoder_sel)
		1:begin
			case(spike_number)
				0:begin codeword <= 'b0; length <= 1; end
				1:begin codeword <= 'b101; length <= 3; end
				2:begin codeword <= 'b110; length <= 3; end
				3:begin codeword <= 'b111; length <= 3; end
				default:begin codeword <= 'b100; length <= 3; end
		
			endcase
		end
		2:begin
			case(spike_number)
				0:begin codeword <= 'b01; length <= 2; end
				1:begin codeword <= 'b10; length <= 2; end
				2:begin codeword <= 'b11; length <= 2; end
				3:begin codeword <= 'b000; length <= 3; end
				default:begin codeword <= 'b001; length <= 3; end
		
			endcase
		end
		
		default: begin
			case(spike_number)
				0:begin codeword <= 'b1; length <= 1; end
				1:begin codeword <= 'b01; length <= 2; end
				2:begin codeword <= 'b001; length <= 3; end
				3:begin codeword <= 'b0; length <= 4; end
				default:begin codeword <= 'b0001; length <= 4; end

			endcase
		end
		endcase
	else begin
		codeword <= 'bx;
		length <= 'bx;
	end
	

	
	
end
endmodule
