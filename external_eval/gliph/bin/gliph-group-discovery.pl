#!/usr/bin/env perl
use Getopt::Long;
use vars qw($rootdir);
BEGIN{
  use File::Basename;
  use Cwd 'abs_path';
  $rootdir = dirname(dirname(abs_path($0)));
};
use lib "$rootdir/lib";
use strict;
use warnings;
  
print "\nGLIPH - Grouping Lymphocyte Interfaces by Paratope Hotspots\n"; 
print "GLIPH-1.0rc\n";
print "\nAn algorithm for statistical clustering of adaptive repertoire convergence\n"; 
print "Contact: Jacob Glanville (jakeg\@stanford.edu)\n\n";

################################## Arguments ###################################
 
my ($textfile,$refdb,$globalConverganceCutoff,
    $localConvergenceMotifList,$samplingDepth,$motif_minp,
    $motif_min_foldchange,$kmer_mindepth,
    $global,$local,$make_depth_fig,$discontinuous,$positional_motifs,
    $cdr3len_stratify,$vgene_constrain,$public_tcrs,
    $use_structural_boundaries,$length_stratification)=GatherOptions();


################################## Inputs ######################################

my %unique_h3s=();				     # convert all inputs to
						     # hash of unique CDR3s.
                                                     # if multiple subjects contribute
						     # then optionally that motif
						     # can be enriched
 
my $corename="cag";                                  # corename prefix for
						     # output files

if(-f $textfile){				     # if textfile input, load
  %unique_h3s=loadtextDB($textfile);		     # directly into unique_h3s
  $corename=$textfile;
  $corename=~s/.txt$//;
}

my %refdb_hash_h3s=();				     # initialize the reference
my @refdb_unique_h3s=();			     # database data structures
my $refdb_nseqs=0;				     # but keep empty unless
my %refdb_kmers=();			             # performing simulations
my @testset_redundant_nseqs=loadtextDBRedundantArray($textfile); 
						     # loading number of redundant clones into an array
my $nseqs = scalar(keys %unique_h3s);		     # count the number of CDR3s
my $nseqs_redundant = scalar(@testset_redundant_nseqs);

      # load the reference database
      if(scalar(@refdb_unique_h3s)>0){
        print "  reference db has already been loaded\n";
      }else{
        print "  loading reference db        $refdb\n";
        %refdb_hash_h3s=loadvdjfastaDB($refdb);
        @refdb_unique_h3s=keys %refdb_hash_h3s;
        $refdb_nseqs=scalar(@refdb_unique_h3s);
      }

################################## Outputs #####################################

my $kmer_sim_log            = $corename . "-kmer_resample_" . $samplingDepth . "_log.txt";
my $clone_network           = $corename . "-clone-network.txt";
my $convergence_table       = $corename . "-convergence-groups.txt";
my $global_similarity_table = $corename . "-global-similarity.txt";

unless(-f $localConvergenceMotifList){
  $localConvergenceMotifList = $corename . "-kmer_resample_" . $samplingDepth . "_minp" 
                             . $motif_minp . "_ove" . $motif_min_foldchange . ".txt";
}

################################## Analysis ####################################

if($make_depth_fig){
  print "  simulated stochastic resampling of depth $nseqs unique seqs out of $nseqs_redundant seqs from $refdb\n";
  getGlobalConvergenceDistribution(\@testset_redundant_nseqs,"Sample",$global_similarity_table);

  for(my $s=0;$s<10;$s++){
    my @random_subsample=randomSubsample(\@refdb_unique_h3s,$nseqs_redundant,$length_stratification,\@testset_redundant_nseqs);
    getGlobalConvergenceDistribution(\@random_subsample,"Sim$s",$global_similarity_table);
  }
}

if($global){
  print "Establishing global convergence significance cutoff\n";
  print "=====================================================================\n";

  if( $globalConverganceCutoff eq ""){
    print "Evaluating depth of input sample in determining distance cutoff\n";
#depth dist1 dist2 dist3
#25   0.0016  0.00239 0.012
#50   0.0004  0.0032  0.0274
#75   0.0008  0.00626 0.03133
#100  0.00059 0.0076  0.04969
#125  0.00176 0.01024 0.06328 (0.01 cutoff for dist 2)
#150  0.00119 0.0122  0.06913
#200  0.00225 0.01725 0.0896
#300  0.00333 0.02553 0.11913
#400  0.00457 0.03215 0.14322
#500  0.00449 0.03854 0.16704
#600  0.00695 0.0478  0.18053 (0.05 cutoff for dist 2 - maybe 650)
#700  0.00671 0.05374 0.20252
#800  0.00737 0.05976 0.21313
#900  0.00873 0.06694 0.22701
#1000 0.00996 0.07113 0.23726 (0.01 cutoff for dist 1)
#2000 0.0193  0.12085 0.29814
#3000 0.02916 0.15676 0.3221
#4000 0.03842 0.18685 0.32965
#5000 0.04548 0.2138 0.32751   (0.05 cutoff for dist 1  - 5500)
#6000 0.05694 0.23055 0.32172
    # nseqs<125 = 0.01, nseqs<650 = 0.05, nseqs<1000 = 0.01, nseqs<5500 = 0.05
    if($nseqs<125){
      $globalConverganceCutoff=2; 
    }else{
      $globalConverganceCutoff=1;
    }
  }else{
    print "Received custom global convergence significance cutoff ";
    print "  $globalConverganceCutoff as input\n";
  }
}

my @local_motifs=();
if($local){
  print "\n";
  print "Preparing local convergence motifs (3mers,4mers,xx.x,x.xx)\n";
  print "=====================================================================\n";
  # if $localConvergenceMotifList is not set, then run motif convergence analysis by resampling
  unless( -f $localConvergenceMotifList){
    print "  simulated stochastic resampling of depth $nseqs from reference db\n";

    # run the motif search, length 2 3 4, x.x, x..x
    print "  loading kmers from sample...\n";
    my @unique_cdrs=keys %unique_h3s; # monkey
    my %sample_kmers=getAllKmerLengths(\@unique_cdrs,$discontinuous,$use_structural_boundaries);
    my @sample_kmer_array=keys %sample_kmers;
    print "  kmers obtained:             " . scalar(@sample_kmer_array) . "\n";
    my @sample_highcount_kmer_array = getMinDepthKmerArray(\%sample_kmers,$kmer_mindepth);
    print "  mindepth>=$kmer_mindepth kmers obtained: " . scalar(@sample_highcount_kmer_array) . "\n"; 

    # print kmer scores to the kmer subsampling log
    appendToKmerLog(\@sample_highcount_kmer_array,\%sample_kmers,"Discovery",$kmer_sim_log);
 
    # load the reference database
    if(scalar(@refdb_unique_h3s)>0){
      print "  reference db has already been loaded\n";
    }else{
      print "  loading reference db        $refdb\n";
      %refdb_hash_h3s=loadvdjfastaDB($refdb);
      @refdb_unique_h3s=keys %refdb_hash_h3s;
      $refdb_nseqs=scalar(@refdb_unique_h3s);
      print "  Getting reference db kmers\n";
      %refdb_kmers=getAllKmerLengths(\@refdb_unique_h3s,$discontinuous,$use_structural_boundaries);
    } 

    # for elevated motifs, run 1000 simulations to determine how often they are
    # as elevated as observed from a random sampling
    print "Subsampling (depth $nseqs): ";
    my $iter=0;
    for(my $sim=0;$sim<$samplingDepth;$sim++){
      my @random_subsample=randomSubsample(\@refdb_unique_h3s,$nseqs,$length_stratification,\@testset_redundant_nseqs);
      my %refdb_subsample_kmers=getAllKmerLengths(\@random_subsample,$discontinuous,$use_structural_boundaries);
      appendToKmerLog(\@sample_highcount_kmer_array,\%refdb_subsample_kmers,"sim-$sim",$kmer_sim_log);
      if($iter>49){
        print "\n";
        print "Subsampling (depth $nseqs): ";
        $iter=0;
      }
      print "#";
      $| = 1;
      $iter++;
    }
    print "\n";
    analyzeKmerLog($kmer_sim_log,$localConvergenceMotifList,$samplingDepth,
                   $nseqs,$motif_min_foldchange,$motif_minp,\%refdb_kmers,$refdb_nseqs);
 
  }else{
    print "Received local convergence motifs file\n";
    print "  $localConvergenceMotifList as input\n";
  }


  # load motifs from list
  open(FILE,$localConvergenceMotifList);
  #my @local_motifs=();
  my @motif_file_lines=<FILE>;
  chomp(@motif_file_lines);
  close(FILE); 
  for(my $x=0;$x<scalar(@motif_file_lines);$x++){
    print $motif_file_lines[$x] . "\n";
    my @fields=split(/\t/,$motif_file_lines[$x]);
    push @local_motifs,$fields[0];
  }
  print "   Sourced " . scalar(@local_motifs) . " significant motifs\n";
  print "\n\n";
}

print "Global and local clustering of immune interfaces\n";
print "=====================================================================\n";

if(-f $clone_network){
  print "  Clone network already calculated\n";
}else{
  # conduct clustering
  open(NETWORK,">$clone_network");
  print "Clustering (depth $nseqs): ";
  my $iter=0;
  my %cluster_assignment=();
  my @h3s=keys %unique_h3s; # monkey
  my %is_networked=(); # a hash of clones that have been networked. To be used
                       # for cleaning up the un-networked clones
  # for every sequence
  for(my $x=0;$x<$nseqs;$x++){
    # output
    if($iter>49){
      print "\n";
      print "Clustering (depth $nseqs): ";
      $iter=0;
    }
    print "#";
    $| = 1;
    $iter++;

    if(length($h3s[$x])>7){
      for(my $y=($x+1);$y<$nseqs;$y++){  # compare to sequences that haven't been compared to yet
        if(length($h3s[$y])>7){
          if(length($h3s[$x]) == length($h3s[$y])){
            my $dist = getContactDist($h3s[$x],$h3s[$y],$use_structural_boundaries);
            if($dist<=$globalConverganceCutoff){
              $is_networked{$h3s[$x]}=1;
              $is_networked{$h3s[$y]}=1;
              print NETWORK $h3s[$x] . "\t" . $h3s[$y] . "\tglobal\n";
            }
          }
          # LOCAL SEARCH
          # for each motif. if motif is found in both sequences
          for(my $m=0;$m<scalar(@local_motifs);$m++){
            my $h3a=$h3s[$x];
            my $h3b=$h3s[$y];
            if($use_structural_boundaries){
              if(  ($h3s[$x]=~m/...$local_motifs[$m].../) 
                 and 
                ($h3s[$y]=~m/...$local_motifs[$m].../)){
                $is_networked{$h3s[$x]}=1;
                $is_networked{$h3s[$y]}=1;
                print NETWORK $h3s[$x] . "\t" . $h3s[$y] . "\tlocal\n";
              }
            }else{
              if(  ($h3s[$x]=~m/$local_motifs[$m]/)
                 and
                ($h3s[$y]=~m/$local_motifs[$m]/)){
                $is_networked{$h3s[$x]}=1;
                $is_networked{$h3s[$y]}=1;
                print NETWORK $h3s[$x] . "\t" . $h3s[$y] . "\tlocal\n";
              }
            }
          }
        }
      }
    }
  }
  # clean up the network by printing all the residual singletons
  my @unique_cdrs=keys %unique_h3s; # monkey
  for(my $i=0;$i<@unique_cdrs;$i++){
    if(!defined($is_networked{$unique_cdrs[$i]})){
      print NETWORK $unique_cdrs[$i] . "\t" . $unique_cdrs[$i] . "\tsingleton\n";
    }
  }

  # ok, now close the network
  close(NETWORK);
  print "\n";
}

# last part, generating lists of clones in each convergence group
# open the network file, iterate until complete
# if a vdjfasta file, gather those details as well 
print "Analysing network to report convergence groups\n";
open(NETWORK,$clone_network);
my @lines=<NETWORK>;
chomp(@lines);
close(NETWORK);

my %unique_networked_clones=();
my %unique_pairs=();
my %all_connections_per_clone=();
for(my $x=0;$x<scalar(@lines);$x++){
  my($clone1,$clone2,$interaction)=split(/\t/,$lines[$x]);
  $unique_networked_clones{$clone1}=1;
  $unique_networked_clones{$clone2}=1;
  $unique_pairs{$clone1 . "-" . $clone2}=1;
  if(defined($all_connections_per_clone{$clone1})){
    $all_connections_per_clone{$clone1} .= " " . $clone2;
  }else{
    $all_connections_per_clone{$clone1} = $clone2;
  }
  if(defined($all_connections_per_clone{$clone2})){
    $all_connections_per_clone{$clone2} .= " " . $clone1;
  }else{
    $all_connections_per_clone{$clone2} = $clone1;
  }
}
my @unique_pair_list=keys %unique_pairs;
print "Identified " . scalar(@unique_pair_list) . " unique interactions\n";

my @unique_cdrs=keys %unique_networked_clones;
my %convergence_groups=();
my %is_assigned=();

open(CONVERGE,">$convergence_table");
for(my $x=0;$x<scalar(@unique_cdrs);$x++){
  unless(defined($is_assigned{$unique_cdrs[$x]})){
    #print "Initialize Convergence Group " . $unique_cdrs[$x] . ": ";
    $convergence_groups{$unique_cdrs[$x]}=$unique_cdrs[$x];
    $is_assigned{$unique_cdrs[$x]}=1; 

    # iterate until no new clones are found. recursion?
    my %current_clones=();
    my %convergence_group_clone_hash=recursiveSingleLinkage($unique_cdrs[$x],\%current_clones,\%all_connections_per_clone);
    my @convergence_group_clone_list=keys %convergence_group_clone_hash;
    print CONVERGE scalar(@convergence_group_clone_list) . "\tCRG-" . $unique_cdrs[$x] 
          . "\t" . $convergence_group_clone_list[0];
    print scalar(@convergence_group_clone_list) . "\tCRG-" . $unique_cdrs[$x]
          . "\t" . $convergence_group_clone_list[0];
    $is_assigned{$convergence_group_clone_list[0]}=1;
    
    for(my $c=1;$c<scalar(@convergence_group_clone_list);$c++){
      print CONVERGE " " . $convergence_group_clone_list[$c];
      print " " . $convergence_group_clone_list[$c];
      $is_assigned{$convergence_group_clone_list[$c]}=1;
    }
    print CONVERGE "\n";
    print "\n";
    # print out the list of clones
    for(my $c=0;$c<scalar(@convergence_group_clone_list);$c++){
      print "\t" . $convergence_group_clone_list[$c] . "\n";
    }
  }
}



############################### subroutines ###############################

sub motifMinFoldPerCounts {
  my($motif_count)=@_;
  # minimum fold-change for the motif to count
  # there is a motif count to fold-change relationship
  if($motif_count == 2){
    return 1000;
  }elsif($motif_count == 3){
    return 100;
  }elsif($motif_count == 4){
    return 10;
  }elsif($motif_count > 4){
    return 10;
  }
}

sub recursiveSingleLinkage {
  my($seed_clone,$current_clones,$all_connections_per_clone)=@_;

  $$current_clones{$seed_clone}=1;

  # grab this clones connections. If any of them are new, grab their 
  # connections and add them to the pile
  my @redundant_clones=split(/ /,$$all_connections_per_clone{$seed_clone}); 
  my %new_unique_clones=();
  for(my $x=0;$x<scalar(@redundant_clones);$x++){
    $new_unique_clones{$redundant_clones[$x]}=1;
  } 
  my @new_unique_clone_array=keys %new_unique_clones;

  # recurse if any new unique clones are found
  for(my $x=0;$x<scalar(@new_unique_clone_array);$x++){
    # if this is a new clone
    if(!defined($$current_clones{$new_unique_clone_array[$x]})){
      my %get_new_unique_clones=recursiveSingleLinkage($new_unique_clone_array[$x],\%$current_clones,\%$all_connections_per_clone);
      %$current_clones=(%$current_clones,%get_new_unique_clones); 
    }  
  }
  return %$current_clones;
}

sub analyzeKmerLog {
  my($logfile,$localConvergenceMotifList,$simdepth,$seqspersim,$minfoldchange,$minp,$refdb_kmers,$refdb_nseqs)=@_;

  open(LOG,">$localConvergenceMotifList");

  open(FILE,$logfile);
  my @lines=<FILE>;
  chomp(@lines);
  close(FILE);
 
  #my $simdepth=scalar(@lines) - 2;

  my @motifs=split(/ /,$lines[0]);
  my @discovery_sample_counts=split(/ /,$lines[1]);
 
  print LOG "Motif\tCounts\tavgRef\ttopRef\tOvE\tp-value\n"; 
  print "Motif\tCounts\tavgRef\ttopRef\tOvE\tp-value\n";

  for(my $m=1;$m<scalar(@motifs);$m++){

    # get number of simulations at the level observed in discovery sample
    # get highest
    # get average (median?)
    my $highest=0;
    my $average=0;
    my $odds_as_enriched_as_discovery=0;
    for(my $sim=2;$sim<scalar(@lines);$sim++){
      my @fields=split(/ /,$lines[$sim]);
      if($fields[$m]>=$discovery_sample_counts[$m]){
        $odds_as_enriched_as_discovery += 1/$simdepth;
      }
      if($fields[$m]>$highest){
        $highest=$fields[$m];
      }
      $average += ($fields[$m]/(scalar(@lines)-2));
    }

    # get observed vs expected
    my $ove = 0;
    if($average>0){
      $ove=$discovery_sample_counts[$m]/$average;
    }else{
      # if found 0, assume you just missed it. pseudocount of 1
      $ove = 1 / ($simdepth * $seqspersim);
      #$ove=">" . $discovery_sample_counts[$m];
    }
    $ove=( int($ove * 1000) / 1000);   
    $average=( int($average * 100) / 100);
    # calculate fisher's exact by generating a confusion matrix
    # of (counts in $seqspersim) vs ( $$refdb_kmers{kmer} in $refdb_nseqs) 
    #               motif    !motif
    #  discovery    n11      n12   | n1p
    #  refdb        n21      n22   | n2p
    #              --------------
    #               np1      np2   npp

    # 3 100 3 100
    #my $n11 = $discovery_sample_counts[$m];
    #my $n1p = $seqspersim;
    #my $np1 = $discovery_sample_counts[$m];
    #if(defined($$refdb_kmers{$motifs[$m]})){
    #  $np1 += $$refdb_kmers{$motifs[$m]};
    #}
    #my $npp = $seqspersim + $refdb_nseqs;
    #print "$n11 $n1p $np1 $npp | $refdb_nseqs\n";
    #my $left_value = calculateStatistic( n11=>$n11,
    #                                     n1p=>$n1p,
    #                                     np1=>$np1,
    #                                     npp=>$npp);
  
    if($odds_as_enriched_as_discovery<$minp){
      if($odds_as_enriched_as_discovery == 0){
        $odds_as_enriched_as_discovery=(1/$simdepth);
      }
      my $this_minfoldchange=motifMinFoldPerCounts($discovery_sample_counts[$m]);
      # if($ove>=$minfoldchange){
      if($ove>=$this_minfoldchange){ #$minfoldchange){
        print LOG        $motifs[$m] 
          . "\t" . $discovery_sample_counts[$m] 
          . "\t" . $average
          . "\t" . $highest
          . "\t" . $ove
          . "\t" . $odds_as_enriched_as_discovery 
          . "\n"; #. "\tfisher=$left_value\n";
        print        $motifs[$m]
          . "\t" . $discovery_sample_counts[$m]
          . "\t" . $average
          . "\t" . $highest
          . "\t" . $ove
          . "\t" . $odds_as_enriched_as_discovery
          . "\n";
      }
    }
  }
  close(LOG);
}

sub appendToKmerLog {
  my($sample_highcount_kmer_array,$sample_kmers,$label,$kmer_sim_log)=@_;
  if($label eq "Discovery"){
    open(LOG,">$kmer_sim_log");			  
    print LOG "Sample";
    for(my $x=0;$x<scalar(@$sample_highcount_kmer_array);$x++){
      print LOG " " . $$sample_highcount_kmer_array[$x];
    }
    print LOG "\n";
    close LOG
  }

  open(LOG,">>$kmer_sim_log");

  print LOG $label;
  for(my $x=0;$x<scalar(@$sample_highcount_kmer_array);$x++){
    if(defined($$sample_kmers{$$sample_highcount_kmer_array[$x]})){
      print LOG " " . $$sample_kmers{$$sample_highcount_kmer_array[$x]};
    }else{
      print LOG " 0";
    }
  }
  print LOG "\n";
  close(LOG);
}

sub getGlobalConvergenceDistribution {
  my($sequences,$samplename,$global_similarity_table)=@_;

  open(FILE,">>$global_similarity_table");

  my @counts=(0,0,0,0,0, 0,0,0,0,0,
              0,0,0,0,0, 0,0,0,0,0,
              0,0,0,0,0, 0,0,0,0,0,
              0,0,0,0,0, 0,0,0,0,0); 
  for(my $x=0;$x<scalar(@$sequences);$x++){
    my $distance=length($$sequences[$x]);
    for(my $y=0;$y<scalar(@$sequences);$y++){
      if($x != $y){
        if(length($$sequences[$x]) == length($$sequences[$y])){
          my $this_dist=getHammingDist($$sequences[$x],$$sequences[$y]);
          if($this_dist<$distance){
            $distance=$this_dist;
          }
        } 
      }
    }
    $counts[$distance]++;
  }
  
  # print the result
  print $samplename;
  print FILE $samplename;
  for(my $x=0;$x<13;$x++){
    print "\t" . $counts[$x];
    print FILE "\t" . $counts[$x];
  }
  print "\n";
  print FILE "\n";
  return @counts; 

  close(FILE);
}

sub getHammingDist {
  my($seq1,$seq2)=@_;
  my @chars1=split(/ */,$seq1);
  my @chars2=split(/ */,$seq2);

  my $mismatch_columns=0;

  for(my $c=0;$c<scalar(@chars1);$c++){
    if($chars1[$c] ne $chars2[$c]){
      $mismatch_columns++;
    }
  }
  return $mismatch_columns;
}

sub getMinDepthKmerArray {
  my($kmer_hash,$mindepth)=@_;
  my @all_kmers=keys %$kmer_hash;
  my @selected_kmers=();
  for(my $x=0;$x<scalar(@all_kmers);$x++){
    if($$kmer_hash{$all_kmers[$x]}>=$mindepth){
      push @selected_kmers,$all_kmers[$x];
    }   
  }
  return @selected_kmers;
}

sub loadtextDBRedundantArray {
  my($textfile)=@_;
  open(FILE,$textfile);
  my @lines=<FILE>;
  chomp(@lines);
  close(FILE);
  return @lines;
}

sub loadtextDB {
  my($textfile)=@_;
  open(FILE,$textfile);
  my @lines=<FILE>;
  chomp(@lines);
  close(FILE);

  my %unique_h3s=(); # monkey
  #addToHash(\%unique_h3s,$lines[$x]);

  for(my $x=0;$x<scalar(@lines);$x++){
    if($lines[$x]=~m/^C[AC-WY][AC-WY][AC-WY][AC-WY]*F/){
      my @fields=split(/\t/,$lines[$x]);
      addToHash(\%unique_h3s,$fields[0]);
    }
  }
  return %unique_h3s;
}

sub loadvdjfastaDB {
  my($fasta)=@_;
  if(-f $fasta){
    open(FILE,$fasta);
    my @lines=<FILE>;
    chomp(@lines);
    close(FILE);
    my $count=0;
    my %refdb_hash_h3s=();
    for(my $x=0;$x<scalar(@lines);$x++){
      if($lines[$x]=~m/>/){
        $count++;
        my @fields=split(/;/,$lines[$x]);
        my $h3=$fields[4];
        addToHash(\%refdb_hash_h3s,$h3);
      }
    }
    return %refdb_hash_h3s;
  }else{
    print "loadvdjfastaDB failed: file \"$fasta\" not found\n";
    exit;
  }
}

sub getContactDist {
  my($seq1,$seq2,$use_structural_boundaries)=@_;
  $seq1=uc($seq1);
  $seq2=uc($seq2);

  # remove the first three and last three characters
  if($use_structural_boundaries){
    $seq1=~s/^...//;
    $seq1=~s/...$//;
    $seq2=~s/^...//;
    $seq2=~s/...$//;
  }


  my @chars1=split(/ */,$seq1);
  my @chars2=split(/ */,$seq2);

  my $alignable_columns=0;
  my $mismatch_columns=0;
 
  for(my $c=0;$c<scalar(@chars1);$c++){
    if($chars1[$c] =~ m/[A-Z]/){
      if($chars2[$c] =~ m/[A-Z]/){
        if($chars1[$c] ne $chars2[$c]){
          $mismatch_columns++;
        }
        $alignable_columns++;
      }
    }
  }
  if($alignable_columns == 0){
    return(0);
  }else{
    return($mismatch_columns); #,(($alignable_columns - $mismatch_columns)/$alignable_columns),$alignable_columns);
  }
}

sub randomSubsample {
  my($array,$depth,$length_stratification,$testset_array)=@_;
  my @id_array=();
  my @random_subsample=();
 
  unless(defined($depth)){
    $depth=scalar(@$array);
  }
  if($depth>scalar(@$array)){
    $depth=scalar(@$array);
  }

  fisher_yates_shuffle(\@$array);

  if($length_stratification){
    #print "Running length stratification\n";
    # first get the distributions of lengths
    my @cdr3lengths=(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
    for(my $x=0;$x<scalar(@$testset_array);$x++){
      $cdr3lengths[length($$testset_array[$x])]+=1;
    }
    # now for each length go through and collect the appropriate number of cdr3s
    for(my $thiscdr3len=9;$thiscdr3len<23;$thiscdr3len++){
      my $s=0;
      while($cdr3lengths[$thiscdr3len]>0){
        #print "  adding " . $cdr3lengths[$thiscdr3len] . " CDRs of length $thiscdr3len from " . scalar(@$array) . "\t" . $$array[$s] . "\n";
        if(length($$array[$s]) == $thiscdr3len){
          #print "     fuckyeah found one! " . $$array[$s] . "\n";
          push @random_subsample,$$array[$s];
          $cdr3lengths[$thiscdr3len]-=1;
        }
        $s+=1; 
      }
    }  
    #exit; # MONKEY 
  }else{

    for(my $s=0;$s<$depth;$s++){
      push @random_subsample,$$array[$s];
    }
  }
  return @random_subsample;
}

# randomly permutate @array in place
sub fisher_yates_shuffle {
  my ($array) = @_;
  my $i = @$array;
  while ( --$i ) {
    my $j = int rand( $i+1 );
    @$array[$i,$j] = @$array[$j,$i];
  }
}

sub getAllKmerLengths {
  my($unique_cdrs,$discontinuous,$use_structural_boundaries)=@_;
  my %sample_kmers=();

  my %sample_2mers=getKmers(\@$unique_cdrs,2,$use_structural_boundaries);
  my %sample_3mers=getKmers(\@$unique_cdrs,3,$use_structural_boundaries);
  my %sample_4mers=getKmers(\@$unique_cdrs,4,$use_structural_boundaries);

  if($discontinuous){
    my %sample_xxox4mers=getKmers(\@$unique_cdrs,"xx.x",$use_structural_boundaries);
    my %sample_xoxx4mers=getKmers(\@$unique_cdrs,"x.xx",$use_structural_boundaries);
    %sample_kmers=(%sample_2mers,%sample_3mers,%sample_4mers,%sample_xxox4mers,%sample_xoxx4mers);
  }else{ 
    %sample_kmers=(%sample_2mers,%sample_3mers,%sample_4mers); #,%sample_xxox4mers,%sample_xoxx4mers);
  }

  return %sample_kmers;
}

sub getKmers {
  my($h3_array,$kmer_size,$use_structural_boundaries)=@_;
  my %kmer_hash=();
  my $nseqs=scalar(@$h3_array);
  my $mask_type=$kmer_size;
  for(my $s=0;$s<$nseqs;$s++){
    my $seq=$$h3_array[$s];
    if($use_structural_boundaries){
       $seq=~s/^...//;
       $seq=~s/...$//;
    }
   
    my $length = length($seq);
    
    # dealing with x.x, x..x, or x...x
    my $mask=0;
    if($mask_type =~ m/x/){
      $mask=1;
      $kmer_size=length($mask_type);
    }

    # run search
    for(my $p=0;($p+$kmer_size)<=$length;$p++){
      my $kmer = substr($seq,$p,$kmer_size);
      # dealing with x.x, x..x, or x...x
      if($mask){
        my @chars=split(/ */,$kmer);
        if($mask_type eq "xx.x"){
          $chars[2]=".";
        }elsif($mask_type eq "x.xx"){
          $chars[1]=".";
        }
        $kmer=join("",@chars);
      }

      if(defined($kmer_hash{$kmer})){
        $kmer_hash{$kmer}++;
      }else{
        $kmer_hash{$kmer}=1;
      }
    }
  }
  return %kmer_hash;
}



sub addToHash {
  my($hash,$newkey)=@_;
  if(defined($$hash{$newkey})){
    $$hash{$newkey}++;
  }else{
    $$hash{$newkey}=1;
  }
}

sub GatherOptions {
  my $textfile         		= "";
  my $globalConverganceCutoff	= ""; # if not defined, this is discovered
  my $localConvergenceMotifList = ""; # if defined, this test is not performed
  my $refdb            		= "$rootdir/db/tcrab-naive-refdb-pseudovdjfasta.fa";
  my $resampling_depth          = 1000;
  my $motif_min_foldchange      = 10;
  my $motif_minp                = 0.001;
  my $kmer_mindepth             = 3;
  my $global                    = 1; #default allow global
  my $local                     = 1; #default allow local
  my $make_depth_fig            = 0; # generate a depth sampling figure
  my $discontinuous             = 0; # allow discontinuous motifs
  my $positional_motifs         = 0; # fix motif positions N-terminlally
  my $cdr3len_stratify          = 0; # stratify by CDR3 length
  my $vgene_constrain           = 0; # limit to dominant V-gene in each GLIPH cluster   
  my $public_tcrs               = 0; # recount public clones
  my $use_structural_boundaries = 1; # should we ignore the first and last few residues of CDR3?
  my $length_stratification     = 0; # apply length stratification to match length
				     # distribution in test set to naive random pulls

  GetOptions(
     "--tcr=s"           => \$textfile,
     "--refdb=s"         => \$refdb,
     "--gccutoff=s"      => \$globalConverganceCutoff,
     "--motif_file=s"    => \$localConvergenceMotifList,
     "--simdepth=s"      => \$resampling_depth,
     "--lcminp=s"        => \$motif_minp,
     "--lcminove=s"      => \$motif_min_foldchange,
     "--kmer_mindepth=s" => \$kmer_mindepth,
     "--global=s"        => \$global,
     "--local=s"         => \$local,
     "--make_depth_fig=s"=> \$make_depth_fig,
     "--discontinuous=s" => \$discontinuous,
     "--positional_motifs=s" => \$positional_motifs,
     "--cdr3len_stratify=s"  => \$cdr3len_stratify,
     "--vgene_constrain=s"    => \$vgene_constrain,
     "--public_tcrs=s"       => \$public_tcrs,
     "--structboundaries=s"  => \$use_structural_boundaries,
     "--length_stratify=s"   => \$length_stratification
  );

  unless(-f $textfile){
    print "\nUsage: $0\n";
    print "  --tcr=seqs.txt         Tab sep input with col 1 containing CDR3s\n";
    print "  --refdb=vdjfasta.fa    Optional alternative reference database\n";
    print "  --gccutoff=1           Global covergence distance cutoff, or\n";
    print "                         global convergence simulation file.\n";
    print "                         Calculated at runtime if not specified.\n";
    print "  --motif_file=motif.txt Optional local convergence significant\n";
    print "                         motif list (calculated if not specified)\n";
    print "  --simdepth=1000        Simulated resampling depth for\n";
    print "                         non-parametric convergence significance\n";
    print "                         tests\n";
    print "  --lcminp=0.01          Local convergence minimum p-value for\n";
    print "                         significance (0.01 by default)\n";
    print "  --lcminove=10          Local convergence minimum observed vs\n";
    print "                         expected fold change (10 by default).\n";
    print "  --kmer_mindepth=3      Minimum observations of kmer for it to\n";
    print "                         be evaluated\n";
    
    print "  --global=1             Search for global TCR similarity (Default 1)\n"; 
    print "  --local=1              Search for local TCR similarity (Default 1)\n";
    print "  --make_depth_fig=0     Perform repeat random samplings at the test\n";
    print "                         set depth in order to visualize convergence\n";
    print "  --discontinuous=0      Allow discontinuous motifs (Default 0)\n";
    print "  --positional_motifs=0  Restrict motif clustering to a shared position\n";
    print "                         that is fixed from the N-terminal end of CDR3\n";
    print "  --cdr3len_stratify=0   Stratify by shared cdr3length distribution\n";
    print "                         (Default 0)\n";
    print "  --vgene_constrain=0    Stratify by shared V-gene frequency distribution\n";
    print "                         (Default 0)\n";
    print "  --public_tcrs=0        Reward motifs in public TCRs (Default 0)\n";
    print "  --structboundaries=1   Use structural boundaries (Default 1)\n";
    print "  --length_stratify=1    Apply length stratification to match CDR3 length\n";
    print "                         distribution to control set random samplings\n";
    print "                         (default 0)\n";
    exit;
  }
  return ($textfile,$refdb,$globalConverganceCutoff,
          $localConvergenceMotifList,$resampling_depth,$motif_minp,
          $motif_min_foldchange,$kmer_mindepth,
          $global,$local,$make_depth_fig,$discontinuous,$positional_motifs,
          $cdr3len_stratify,$vgene_constrain,$public_tcrs,
          $use_structural_boundaries,$length_stratification);
}

sub stripFastaSuffix {
  my($corename)=@_;
   $corename=~s/.fa$//;
   $corename=~s/.fasta$//;
   $corename=~s/.fna$//;
   return $corename;
}


