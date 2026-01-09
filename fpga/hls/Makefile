# TRACCC FPGA RK4 Propagator - Makefile
#
# Targets:
#   csim    - Build and run C-simulation testbench
#   hls     - Run Vitis HLS synthesis (requires Vitis HLS installation)
#   clean   - Remove build artifacts
#
# Usage:
#   make csim           # Run C-simulation
#   make hls PART=xcv80 # Run HLS synthesis for V80

CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

# Vitis HLS settings (adjust path as needed)
VITIS_HLS ?= vitis_hls
PART ?= xcv80
CLOCK_PERIOD ?= 2.0

# Source files
SRCS = rk4_propagator.cpp
HEADERS = rk4_propagator.hpp
TB_SRC = rk4_propagator_tb.cpp

# Build directory
BUILD_DIR = build

#==============================================================================
# C-Simulation
#==============================================================================

.PHONY: csim
csim: $(BUILD_DIR)/rk4_tb
	@echo "Running C-simulation..."
	@./$(BUILD_DIR)/rk4_tb

$(BUILD_DIR)/rk4_tb: $(TB_SRC) $(SRCS) $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(TB_SRC) $(SRCS) -lm

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

#==============================================================================
# Vitis HLS Synthesis
#==============================================================================

.PHONY: hls
hls: hls_script.tcl
	@echo "Running Vitis HLS synthesis..."
	$(VITIS_HLS) -f hls_script.tcl

hls_script.tcl: $(SRCS) $(HEADERS)
	@echo "Generating HLS synthesis script..."
	@echo "# Auto-generated HLS synthesis script" > $@
	@echo "open_project -reset rk4_propagator_hls" >> $@
	@echo "set_top rk4_propagate_const_bfield" >> $@
	@echo "add_files rk4_propagator.cpp" >> $@
	@echo "add_files rk4_propagator.hpp" >> $@
	@echo "add_files -tb rk4_propagator_tb.cpp" >> $@
	@echo "open_solution -reset solution1" >> $@
	@echo "set_part {$(PART)}" >> $@
	@echo "create_clock -period $(CLOCK_PERIOD) -name default" >> $@
	@echo "config_compile -pipeline_loops 1" >> $@
	@echo "csim_design" >> $@
	@echo "csynth_design" >> $@
	@echo "# cosim_design  # Uncomment for RTL co-simulation" >> $@
	@echo "# export_design -format ip_catalog  # Uncomment to export IP" >> $@
	@echo "exit" >> $@

#==============================================================================
# Synthesis Report
#==============================================================================

.PHONY: report
report:
	@if [ -d rk4_propagator_hls/solution1/syn/report ]; then \
		echo "=== HLS Synthesis Report ==="; \
		cat rk4_propagator_hls/solution1/syn/report/rk4_propagate_const_bfield_csynth.rpt | head -100; \
	else \
		echo "No synthesis report found. Run 'make hls' first."; \
	fi

#==============================================================================
# Clean
#==============================================================================

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
	rm -rf rk4_propagator_hls
	rm -f hls_script.tcl
	rm -f *.log
	rm -f vitis_hls.log

.PHONY: help
help:
	@echo "TRACCC FPGA RK4 Propagator Build System"
	@echo ""
	@echo "Targets:"
	@echo "  csim    - Build and run C-simulation testbench"
	@echo "  hls     - Run Vitis HLS synthesis"
	@echo "  report  - Display HLS synthesis report"
	@echo "  clean   - Remove build artifacts"
	@echo ""
	@echo "Variables:"
	@echo "  PART         - Target FPGA part (default: xcv80)"
	@echo "  CLOCK_PERIOD - Target clock period in ns (default: 2.0)"
	@echo "  VITIS_HLS    - Path to vitis_hls command"
