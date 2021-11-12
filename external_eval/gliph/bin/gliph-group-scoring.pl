#!/usr/bin/env perl
use Getopt::Long;
use vars qw($rootdir);
BEGIN{
  use File::Basename;
  use Cwd 'abs_path';
  $rootdir = dirname(dirname(abs_path($0)));
};
use lib "$rootdir/lib";
#use Text::NSP::Measures::2D::Fisher::left;
use strict;
use warnings;
  
print "\nGLIPH - Grouping Lymphocyte Interfaces by Paratope Hotspots\n"; 
print "\nAn algorithm for statistical clustering of adaptive repertoire convergence\n"; 
print "Contact: Jacob Glanville (jakeg\@stanford.edu)\n\n";

################################## Arguments ###################################
 
my ($convergence_file,$clone_annotation_file,$individual_hlas,$pdepth,$motif_pvalue_file)=GatherOptions();

################################## Inputs ######################################

# load HLA structures
  my @hlas        = ();
  my @patients    = ();
  my %patient_hla = ();
  if(-f $individual_hlas){
    load_hla_hash($individual_hlas,\@hlas,\@patients,\%patient_hla);
    #print "QC the expected HLA data structure once I get some actual breathing room\n";
  }

# load annotation structures
  my @clone_annotation_lines=loadFile($clone_annotation_file);
  #print "QC the expected data structure once I get some actual breathing room\n";
  #print "CDR3b	TRBV	TRBJ	Patient	Group	TRAV	TRAJ	CDR3a\n";

# load convergence groups
  my @convergence_groups=loadFile($convergence_file);

# load background v-gene distribution
 
  # first load default vgene and cdr3 profiles
  my @background_vgene_frequencies=getDefaultBackgroundFreqs("$rootdir/db/tcrb-human.v-freq.txt");
  # if clone_annotation_file exists, use that as the background
  if(-f $clone_annotation_file){
    #@background_vgene_frequencies=getBackgroundVgeneFreqs(\@clone_annotation_lines);
    #my @unique_clones=getUniqueClones(\@clone_annotation_lines);
  } 

# load background CDR3-length distribution
  my @background_cdr3_frequencies=getDefaultBackgroundFreqs("$rootdir/db/tcrb-human.cdr3len-freq.txt");
  # if clone annotation file exists, use that as background
  if(-f $clone_annotation_file){
    #@background_cdr3_frequencies=getBackgroundVgeneFreqs(\@clone_annotation_lines);
    #my @unique_clones=getUniqueClones(\@clone_annotation_lines);
  }

# load motif scores

my %motif_p=loadMotifs($motif_pvalue_file);
my @motif_list=keys %motif_p;

################################## Analysis ####################################

# for each convergence group
for(my $c=0;$c<scalar(@convergence_groups);$c++){
  # 2       CRG-CSVRGGRTNTGELFF     CSARGGRTNTGELFF CSVRGGRTNTGELFF
  my($count,$name,$cdr3_list)=split(/\t/,$convergence_groups[$c]);
  # if this convergence group is size 3 or larger


  if($count>1){
    print "\n\nEvaluating $name ($count members: $cdr3_list)\n";

    # get all the patients
    my @crg_subject_array=getSubjectsInConvergenceGroup($cdr3_list,\@clone_annotation_lines); 
    # get all the unique clones
    my @crg_unique_clones=getUniqueClonesInConvergenceGroup($cdr3_list,\@clone_annotation_lines);

    print "\t" . scalar(@crg_subject_array) . " subjects " . " and " . scalar(@crg_unique_clones) . " clones\n";

    my %motifs_here=();
    for(my $m=0;$m<scalar(@motif_list);$m++){
      # %motif_p
      for(my $clone=0;$clone<scalar(@crg_unique_clones);$clone++){
        if($crg_unique_clones[$clone]=~m/$motif_list[$m]/){
          if(defined($motifs_here{$motif_list[$m]})){
            $motifs_here{$motif_list[$m]}++;
          }else{
            $motifs_here{$motif_list[$m]}=1;
          }
        }
      }      
    }
    my $motif_line="";
    print "Motifs: ";
    my @motif_keys=keys %motifs_here;
    for(my $k=0;$k<scalar(@motif_keys);$k++){
      $motif_line .= " " . $motif_keys[$k] . "(" . $motifs_here{$motif_keys[$k]} . ", " . $motif_p{$motif_keys[$k]} . ")";
    }
    print "Motifs: " . $motif_line . "\n";

    # get v-gene enrichment
    my @crg_Vbs=getUniqueCloneVgenes(\@crg_unique_clones);
    my $Vb_p = calculate_enrichment_p(\@background_vgene_frequencies,\@crg_Vbs,$pdepth);

    # get cdr3 length enrichment
    my @crg_cdr3blens=getUniqueCloneCDR3lens(\@crg_unique_clones);
    my $cdr3blen_p = calculate_enrichment_p(\@background_cdr3_frequencies,\@crg_cdr3blens,$pdepth);

    # get HLA
    my $lowest_hla_score=1;
    my $lowest_hla="";
    for(my $h=0;$h<scalar(@hlas);$h++){
      my $all_patient_count=scalar(@patients);
      my $all_patient_hla_count=count_hla_carriers(\@patients,\%patient_hla,$hlas[$h]);
      my $crg_patient_count=scalar(@crg_subject_array);
      my $crg_patient_hla_count=count_hla_carriers(\@crg_subject_array,\%patient_hla,$hlas[$h]);
      if($crg_patient_hla_count>1){
        # # hla_probability(popsize,clustersize,clusterhla,pophla,simsize)
        my ($number_of_pass_cutoff_draws,$p)=hla_probability($all_patient_count,$crg_patient_count,$crg_patient_hla_count,$all_patient_hla_count,100000);
        print "\t" . $hlas[$h] . "\t($crg_patient_hla_count/$crg_patient_count) vs ($all_patient_hla_count/$all_patient_count))\t$number_of_pass_cutoff_draws\t$p\n";
        if($p<$lowest_hla_score){
          $lowest_hla_score=$p;
        }
        if($p<0.1){
          $lowest_hla .= $hlas[$h] . "($p) ";
        }
      }
    }
  
    # get size significance
    my $size_p = getConvergeceGroupSizeP(scalar(split(/ /,$cdr3_list)));

    # get clonal expansion score
    my $expansion_p=getExpansionP($cdr3_list,\@clone_annotation_lines); 

    # get motif enrichment - which motifs were found here, and what are their values? 
    my $motif_p=0.001;
 
    # printing out the clones
    my @clones=split(/ /,$cdr3_list);
    for(my $x=0;$x<scalar(@clones);$x++){
      for(my $z=0;$z<scalar(@clone_annotation_lines);$z++){
        if($clone_annotation_lines[$z]=~m/^$clones[$x]/){
          print "  " . $clone_annotation_lines[$z] . "\n";
        }
      }
    }

    # print the score report
    print "\nFinal scores:\n";
    print "\tVsegment_p\t$Vb_p\n";
    print "\tcdr3len_p\t$cdr3blen_p\n";
    print "\tlowest_hla\t$lowest_hla_score\t$lowest_hla\n";
    print "\texpansion\t$expansion_p\n";
    print "\tcluster size\t$size_p\n";
    print "\tmotifs\t$motif_p\n";
 
    my $score = ($Vb_p * $cdr3blen_p * $lowest_hla_score * $expansion_p * $motif_p * $size_p * 64);
    print "\tFINAL SCORE = $score\n";

    my $unique_subjects = scalar(@crg_subject_array);
    my $unique_clones   = scalar(@crg_unique_clones);
    print "Name\tCDR3s\tSubjects\tClones\tCRG_Score\tVb_p\tCDR3_p\tHLA_p\tExpansion_p\tMotif_p\tSize_p\tHLA\tMotifs\n";
    print "$name\t$count\t$unique_subjects\t$unique_clones\t$score";
    print "\t$Vb_p\t$cdr3blen_p\t$lowest_hla_score\t$expansion_p\t$motif_p\t$size_p\t$lowest_hla\t$motif_line\n";
  }
}

#sub hla_probability {
#sub calculate_enrichment_p {
#sub get_simpson_index {
#sub count_hla_carriers {
#sub load_hla_hash {

################################## subroutines #################################

sub loadMotifs {
  my($file)=@_;
  
  my %motif_p=();
  my @lines=loadFile($file);

  for(my $x=1;$x<scalar(@lines);$x++){
    my ($motif,$counts,$avgref,$topref,$ove,$p)=split(/\t/,$lines[$x]);
    $motif_p{$motif}=$p;
  }
  return %motif_p;
}

sub getExpansionP {
  my($cdr3_list,$clone_annotation_lines)=@_;
  # expanded clones count as 1. Non-expanded count as 0  
  my $crg_count_per_clone=getCountPerCloneInConvergenceGroup($cdr3_list,\@$clone_annotation_lines);
  my $crg_cdr3_count=scalar(split(/ /,$cdr3_list));
 
  # how often can you get a better score than this, randomly sampling from the annotation reference file
  my $equal_or_surpass_score_counts=0;
  
  for(my $s=0;$s<1000;$s++){
    my @subsample_list=randomSubsample(\@clone_annotation_lines,$crg_cdr3_count);
    my $this_score=0;
    for(my $i=0;$i<scalar(@subsample_list);$i++){
      my($CDR3b,$TRBV,$TRBJ,$CDR3a,$TRAV,$TRAJ,$patient,$counts)=split(/\t/,$subsample_list[$i]);
      if($counts eq "Counts"){ # this is for handling a lazy bug
        $counts=0;
      }
      if($counts>1){
        $this_score+=1;
      }
    }
    $this_score=$this_score/$crg_cdr3_count;
 
    if($this_score>=$crg_count_per_clone){
      $equal_or_surpass_score_counts+=1;
    }
  }

  if($equal_or_surpass_score_counts==0){
    $equal_or_surpass_score_counts=1;
  }
  my $expansion_p=$equal_or_surpass_score_counts/1000;
  return $expansion_p;
}

sub getConvergeceGroupSizeP {
  my($unique_clone_count)=@_;
  # observed rates in naive simulations
  my %counts2scores = (
                        1,0.954980692,
                        2,0.029106402,
                        3,0.006190808,
                        4,0.002347221,
                        5,0.001456437,
                        6,0.001212588,
                        7,0.000862264,
                        8,0.000829941,
                        9,0.000476289,
                        10,0.000313723,
                        11,0.000240521,
                        12,0.00015401,
                        13,7.70048E-05,
                        14,7.41528E-05,
                        15,9.50677E-05,
                        16,8.55609E-05,
                        17,9.50677E-05,
                        18,8.55609E-05,
                        19,0.000111229,
                        20,8.08075E-05,
                        21,9.60184E-05,
                        22,7.51035E-05,
                        23,8.46103E-05,
                        24,8.36596E-05,
                        25,5.22872E-05,
                        26,3.80271E-05,
                        27,2.56683E-05,
                        28,3.99284E-05,
                        29,3.99284E-05,
                        30,3.04217E-05,
                        31,2.37669E-05,
                        32,2.85203E-05,
                        33,3.70764E-05,
                        34,3.42244E-05,
                        35,3.04217E-05,
                        36,3.32737E-05,
                        37,3.04217E-05,
                        38,3.32737E-05,
                        39,3.99284E-05,
                        40,2.9471E-05,
                        41,3.04217E-05,
                        42,3.2323E-05,
                        43,2.6619E-05,
                        44,1.14081E-05,
                        45,1.42602E-05,
                        46,1.52108E-05,
                        47,1.52108E-05,
                        48,1.71122E-05,
                        49,2.18656E-05,
                        50,1.42602E-05);
  if(defined($counts2scores{$unique_clone_count})){
    return $counts2scores{$unique_clone_count};
  }else{
    return 1E-05;
  }
}

sub getDefaultBackgroundFreqs {
  my($file)=@_;
  
  my @lines=loadFile($file);
 
  my @freq_array=();
  for(my $x=0;$x<@lines;$x++){
    my($name,$freq)=split(/\t/,$lines[$x]);
    push @freq_array,$freq;
  }
  return @freq_array;
}


sub getSubjectsInConvergenceGroup {
  my($cdr3_list,$clone_annotation_lines)=@_;
 
  my %crg_subjects=();
    
  # get all the unique clones
  # from all the unique clones, get the V-gene distributions, CDR3 length distributions, expansions 
  my @clones=split(/ /,$cdr3_list);
  for(my $x=0;$x<scalar(@clones);$x++){
    for(my $z=0;$z<scalar(@clone_annotation_lines);$z++){
      if($clone_annotation_lines[$z]=~m/^$clones[$x]/){
        my($CDR3b,$TRBV,$TRBJ,$CDR3a,$TRAV,$TRAJ,$patient,$counts)=split(/\t/,$clone_annotation_lines[$z]);
        $crg_subjects{$patient}=1;
      }
    }
  }
  my @crg_subject_array=keys %crg_subjects;
  return @crg_subject_array;
}

sub getCountPerCloneInConvergenceGroup {
  my($cdr3_list,$clone_annotation_lines)=@_;

  my $total=0;
  my @clones=split(/ /,$cdr3_list);
  for(my $x=0;$x<scalar(@clones);$x++){
    for(my $z=0;$z<scalar(@clone_annotation_lines);$z++){
      if($clone_annotation_lines[$z]=~m/^$clones[$x]/){
        my($CDR3b,$TRBV,$TRBJ,$CDR3a,$TRAV,$TRAJ,$patient,$counts)=split(/\t/,$clone_annotation_lines[$z]);
        if($counts eq "Counts"){
          $counts=0;
        }
        if($counts>1){
          $total+=1;
        }
      }
    }
  }
  my $count_per_clone = $total / scalar(@clones);
  return $count_per_clone;
}

sub getUniqueClonesInConvergenceGroup {
  my($cdr3_list,$clone_annotation_lines)=@_;

  my %crg_clones=();

  # get all the unique clones
  # from all the unique clones, get the V-gene distributions, CDR3 length distributions, expansions 
  my @clones=split(/ /,$cdr3_list);
  for(my $x=0;$x<scalar(@clones);$x++){
    for(my $z=0;$z<scalar(@$clone_annotation_lines);$z++){
      if($$clone_annotation_lines[$z]=~m/^$clones[$x]/){
        my($CDR3b,$TRBV,$TRBJ,$CDR3a,$TRAV,$TRAJ,$patient,$counts)=split(/\t/,$$clone_annotation_lines[$z]);
        $crg_clones{$patient . "_" . $TRBV . "_" . $TRBJ . "_" . $CDR3b}=1;
      }
    }
  }
  my @crg_clone_array=keys %crg_clones;
  return @crg_clone_array;
}

sub getUniqueCloneVgenes {
  my($crg_clone_array)=@_;
  my @list=();
  for(my $x=0;$x<scalar(@$crg_clone_array);$x++){
    my($patient,$TRBV,$TRBJ,$CDR3b)=split(/_/,$$crg_clone_array[$x]);
    push @list,$TRBV;
  }
  return @list;
}

sub getUniqueCloneCDR3lens {
  my($crg_clone_array)=@_;
  my @list=();
  for(my $x=0;$x<scalar(@$crg_clone_array);$x++){
    my($patient,$TRBV,$TRBJ,$CDR3b)=split(/_/,$$crg_clone_array[$x]);
    push @list,length($CDR3b);
  }
  return @list;
}

sub getBackgroundVgeneFreqs {
  my($clone_annotation_lines)=@_;
  my @unique_clones=getUniqueClones(\@$clone_annotation_lines);
  my @vgene_freqs=(); # all adds up to 1: frequencies
  
}

sub getUniqueClones {
  my($clone_annotation_lines)=@_;
  my %crg_clones=();

  # get all the unique clones
  for(my $z=0;$z<scalar(@$clone_annotation_lines);$z++){
    my($CDR3b,$TRBV,$TRBJ,$CDR3a,$TRAV,$TRAJ,$patient,$counts)=split(/\t/,$$clone_annotation_lines[$z]);
    $crg_clones{$TRBV . "_" . $TRBJ . "_" . $CDR3b}=1;
  }
  my @crg_clone_array=keys %crg_clones;
  return @crg_clone_array;
}


sub loadFile {
  my($file)=@_;
  open(FILE,$file);
  my @lines=<FILE>;
  close(FILE);
  chomp(@lines);
  return @lines;
}

sub hla_probability {
  my($N,$k,$x,$A,$S)=@_; 
  #print "popsize N                  = $N\n"; N=22
  #print "cluster k                  = $k\n"; k=5
  #print "observed HLA Xs in cluster = $x\n"; x=4 five DQA1*01:02
  #print "total HLA Xs in population = $A\n"; A=13
  #print "number of simulations      = $S\n"; S=100000
  # so found 4/5 from a background of 13/22
 
  # formally nCk=n!/(k!(n-k)!) 
  # |E| = (4c3 * 4) + (4chose4) = 17

  my $p = 1;        # probability

  # checking my math with a simulator
  my $number_of_pass_cutoff_draws=0;

  for(my $sim=0;$sim<$S;$sim++){
    # this is a selection
    my $people_left=$N;
    my $hla_of_interest_left=$A;
    my $number_of_hla_chosen=0;
    #print "Sim $sim\n";

    for(my $picks=0;$picks<$k;$picks++){
      my $rand = int(rand($people_left));
      if($rand<$hla_of_interest_left){
        #print "\t$rand<$hla_of_interest_left: ";
        $number_of_hla_chosen++;
        $hla_of_interest_left--;
        #print " hla_chosen=$number_of_hla_chosen hla_interest_left=$hla_of_interest_left\n";
      }#else{
      #  print "\t$rand>$hla_of_interest_left - no hit\n";
      #}
         
      $people_left--;
    }

    if($number_of_hla_chosen>=$x){
      $number_of_pass_cutoff_draws++;
    }
    #print "$number_of_hla_chosen\n";
  }

  $p=$number_of_pass_cutoff_draws/$S;

  return($number_of_pass_cutoff_draws,$p); 
}


sub calculate_enrichment_p {
  # generates a p value as probability of surpassing
  # the observed simpson index of diversity as seen in 
  # the current data
  #      _R
  # l = \   p2
  #     /_ 
  #     i=1

  my($freq_array,$test_data,$sims)=@_;
 
  my $depth=scalar(@$test_data);
  my $test_score=get_simpson_index(\@$test_data);
  my @unselected_score_distribution=();
  my $pass_scores=0;

  for(my $s=0;$s<$sims;$s++){
    my @picks=();
    for(my $d=0;$d<$depth;$d++){
      my $pick=biased_random_pick(\@$freq_array);  
      push @picks,$pick;
    }
    my $score=get_simpson_index(\@picks);
    push @unselected_score_distribution,$score;
    if($score>=$test_score){
      $pass_scores++;
    }
  }
  my $p = $pass_scores / $sims;
  if($p == 0){
    $p=1/$sims;
  }
  return $p;
}

sub get_simpson_index {
  my($picklist)=@_;
  # get frequencies
  my %pick_freqs=();
  for(my $x=0;$x<scalar(@$picklist);$x++){
    if(defined($pick_freqs{$$picklist[$x]})){
      $pick_freqs{$$picklist[$x]}+= (1/scalar(@$picklist));
    }else{
      $pick_freqs{$$picklist[$x]}= (1/scalar(@$picklist));
    }
  }
  #ok, now multiply frequencies
  my $score=1;
  my @keys=keys %pick_freqs;
  for(my $k=0;$k<scalar(@keys);$k++){
    $score*=$pick_freqs{$keys[$k]};
  }
  return $score;
}

sub biased_random_pick {
  my($array)=@_;
  my $pick=rand(1);
  my $as_much=0;
  for(my $x=0;$x<scalar(@$array);$x++){
    $as_much+=$$array[$x];
    if($pick<=$as_much){
      return $x;
    }
  }
  return (scalar(@$array) - 1);
}

sub count_hla_carriers {
  my($patient_array,$patient_hla_hash,$hla)=@_;
  my $count=0;
  for(my $p=0;$p<scalar(@$patient_array);$p++){
    if(defined($$patient_hla_hash{$$patient_array[$p] . "_" . $hla})){
      $count++;
    }
  }
  return $count;
}
 
sub load_hla_hash {
  my($hla_file,$hla_array,$patient_array,$patient_hla_hash)=@_;
  open(FILE,$hla_file);
  my @lines=<FILE>;
  chomp(@lines);
  close(FILE);

  my %unique_hlas=();
  my %patient_hlas=();

  for(my $x=0; $x<scalar(@lines); $x++){
    my @fields = split(/\t/,$lines[$x]);
    push @$patient_array,$fields[0];
    for(my $f=1;$f<scalar(@fields);$f++){
      my $hla=$fields[$f];
         $hla=~s/:.*//;
         $unique_hlas{$hla}=1;
         $$patient_hla_hash{$fields[0] . "_" . $hla}=$hla;
    }
  }

  # get unique HLAs
  @$hla_array = keys %unique_hlas;
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

sub randomSubsample {
  my($array,$depth)=@_;
  my @id_array=();
  my @random_subsample=();
 
  unless(defined($depth)){
    $depth=scalar(@$array);
  }
  if($depth>scalar(@$array)){
    $depth=scalar(@$array);
  }

  fisher_yates_shuffle(\@$array);

  for(my $s=0;$s<$depth;$s++){
    push @random_subsample,$$array[$s];
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

sub addToHash {
  my($hash,$newkey)=@_;
  if(defined($$hash{$newkey})){
    $$hash{$newkey}++;
  }else{
    $$hash{$newkey}=1;
  }
}

sub GatherOptions {
  my $convergence_file          = "";
  my $clone_annotation_file     = "";
  my $individual_hlas           = "";
  my $pdepth                    = 10000;
  my $motif_pvalue_file         = "";

  GetOptions(
     "--convergence_file=s"     => \$convergence_file,
     "--clone_annotations=s"    => \$clone_annotation_file,
     "--hla_file=s"             => \$individual_hlas,
     "--p_depth=s"              => \$pdepth,
     "--motif_pval_file=s"      => \$motif_pvalue_file
  );
 
  unless( (-f $convergence_file) and (-f $clone_annotation_file) and (-f $individual_hlas) and (-f $motif_pvalue_file)){
    print "\nUsage: \n";
    print "   --convergence_file=myclone.gliph-convergence.txt\n";
    print "   --clone_annotations=myclone.tab.delim.txt\n";
    print "   --hla_file=mypatient.hlas.txt\n";
    print "   --p_depth=10000\n";
    print "   --motif_pval_file=pval.txt\n";
    print "\n";
    exit;
  }
  return ($convergence_file, $clone_annotation_file, $individual_hlas,$pdepth,$motif_pvalue_file);
}


