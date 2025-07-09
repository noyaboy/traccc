#!/usr/bin/env perl
use strict;
use warnings;

my $dir = "model_out/readable";
my $template = "../../core/include/traccc/fitting/kalman_filter/kalman_int8_gru_gain_predictor_weights.empty.hpp";
my $output   = "../../core/include/traccc/fitting/kalman_filter/kalman_int8_gru_gain_predictor_weights.hpp";

sub read_numbers {
    my ($file) = @_;
    open my $fh, '<', $file or die "Cannot open $file: $!";
    my @vals;
    while (<$fh>) {
        chomp;
        next if /^\s*$/;
        push @vals, $_;
    }
    return @vals;
}

# read weights and biases
my @w0 = read_numbers("$dir/fc1_int8_weight.txt");
my @w1 = read_numbers("$dir/fc2_int8_weight.txt");
my @w2 = read_numbers("$dir/fc3_int8_weight.txt");
my @b0 = read_numbers("$dir/fc1_bias.txt");
my @b1 = read_numbers("$dir/fc2_bias.txt");
my @b2 = read_numbers("$dir/fc3_bias.txt");

# pad fc1 weights to 32x24
my $rows = 32;
my $cols = scalar(@w0) / $rows;
my $padded_cols = 24;
if ($cols != 23) {
    warn "Unexpected fc1 weight count: " . scalar(@w0) . "\n";
}
my @w0_padded;
for my $r (0 .. $rows-1) {
    push @w0_padded, @w0[$r*$cols .. $r*$cols + $cols - 1];
    push @w0_padded, (0) x ($padded_cols - $cols);
}
@w0 = @w0_padded;

# parse quantisation parameters
my %q;
open my $qh, '<', "$dir/qparams.txt" or die "Cannot open qparams.txt";
while (<$qh>) {
    if (/quant_in:\s*scale=([0-9eE+\-\.]+),\s*zero_point=(\d+)/) {
        $q{QuantInScale} = $1;
        $q{QuantInZeroPoint} = $2;
    } elsif (/fc1:\s*scale=([0-9eE+\-\.]+),\s*zero_point=(\d+)/) {
        $q{FC1Scale} = $1;
        $q{FC1ZeroPoint} = $2;
    } elsif (/fc2:\s*scale=([0-9eE+\-\.]+),\s*zero_point=(\d+)/) {
        $q{FC2Scale} = $1;
        $q{FC2ZeroPoint} = $2;
    } elsif (/fc3:\s*scale=([0-9eE+\-\.]+),\s*zero_point=(\d+)/) {
        $q{FC3Scale} = $1;
        $q{FC3ZeroPoint} = $2;
    }
}
close $qh;

# read template
open my $th, '<', $template or die "Cannot open $template";
local $/; my $content = <$th>; close $th;

sub join_int {
    return join(', ', @_);
}
sub join_float {
    return join(', ', map { sprintf('%sf', $_) } @_);
}

my %replace = (
    '@W0@' => join_int(@w0),
    '@W1@' => join_int(@w1),
    '@W2@' => join_int(@w2),
    '@B0@' => join_float(@b0),
    '@B1@' => join_float(@b1),
    '@B2@' => join_float(@b2),
    '@QuantInScale@' => $q{QuantInScale} . 'f',
    '@QuantInZeroPoint@' => $q{QuantInZeroPoint},
    '@FC1Scale@' => $q{FC1Scale} . 'f',
    '@FC1ZeroPoint@' => $q{FC1ZeroPoint},
    '@FC2Scale@' => $q{FC2Scale} . 'f',
    '@FC2ZeroPoint@' => $q{FC2ZeroPoint},
    '@FC3Scale@' => $q{FC3Scale} . 'f',
    '@FC3ZeroPoint@' => $q{FC3ZeroPoint},
);

for my $k (keys %replace) {
    my $v = $replace{$k};
    $content =~ s/\Q$k\E/$v/g;
}

open my $outf, '>', $output or die "Cannot open $output";
print $outf $content;
close $outf;

print "Weights exported to $output\n";