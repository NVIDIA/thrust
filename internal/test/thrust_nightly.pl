#!/usr/bin/env perl

use strict;
use warnings;
use Getopt::Long;
use Cwd;
use Cwd 'abs_path';
use File::Temp;

my %CmdLineOption;
my $retVal;
my $arch = "";
my $build = "debug";
my $filter_list_file = undef;
my $test_list_file = undef;
my $unit_test_list_file = "internal/test/unittest.lst";
my $testname = undef;
my $valgrind_enable = 0;
my $cudamemcheck_enable = 0;
my $tool_checker = "";
my $timeout_min = 15;
my $dvs = 0;
my $os = "";
my $cygwin = "";
my $openmp = 0;
my $config = "";
my $abi = "";     
my $remote = "";
my $remote_server = "";
my $remote_android = "";
my $remote_path = "/data/thrust_testing";

my @unittestlist;
my @skip_gold_verify_list = (
    "thrust.example.discrete_voronoi",
    "thrust.example.sorting_aos_vs_soa",
    "thrust.example.cuda.simple_cuda_streams",
    "thrust.example.cuda.fallback_allocator",
);

if (`uname` =~ m/CYGWIN/) {
    $cygwin = 1;
    $os = "win32";
} elsif ($^O eq "MSWin32") {
    $os = "win32";
} else {
    $os = `uname`;
    chomp($os);
}

if ($os eq "win32") {
    $ENV{'PROCESSOR_ARCHITECTURE'} ||= "";
    $ENV{'PROCESSOR_ARCHITEW6432'} ||= "";
    if ((lc($ENV{PROCESSOR_ARCHITECTURE}) ne "x86") ||
        (lc($ENV{PROCESSOR_ARCHITECTURE}) eq "amd64") ||
        (lc($ENV{PROCESSOR_ARCHITEW6432}) eq "amd64"))
    {
        $arch = "x86_64";
    }
    else {
        $arch = "i686";
    }
} else {
    $arch = `uname -m`;
    chomp($arch);
}

sub Usage()
{
    print STDERR "Usage:     thrust_nightly.pl <options>\n";
    print STDERR "Options:\n";
    print STDERR "  -help                         : Print help message\n";
    print STDERR "  -forcearch <arch>             : i686|x86_64|ARMv7|aarch64 (default: $arch)\n";
    print STDERR "  -forceabi <abi>               : Specify abi to be used for arm (gnueabi|gnueabihf)\n";
    print STDERR "  -forceos <os>                 : win32|Linux|Darwin (default: $os)\n";
    print STDERR "  -build <release|debug>        : (default: debug)\n";
    print STDERR "  -timeout_min <min>            : timeout in minutes for each individual test\n";
    print STDERR "  -filter-list-file <file>      : path to filter file which contains one invocation per line\n";
    print STDERR "  -test-list-file <file>        : path to file which contains one example program or unit test per line\n";
    print STDERR "  -unit-test-list-file <file>   : path to file which contains one unit test per line\n";
    print STDERR "  -testname <test>              : single example or unit test to run\n";
    print STDERR "  -dvs                          : summary for dvs\n";
    print STDERR "  -openmp                       : test OpenMP implementation\n";
    print STDERR "  -remote_server <server>       : test on remote target (uses ssh)\n";
    print STDERR "  -remote_android               : test on remote android target (uses adb)\n";
    print STDERR "  -remote_path                  : path on remote target to copy test files (default: $remote_path)\n";
}

$retVal = GetOptions(\%CmdLineOption,
                     'help'     => sub { Usage() and exit 0 },
                     "forcearch=s" => \$arch,
                     "forceabi=s" => \$abi,
                     "forceos=s" => \$os,
                     "build=s" => \$build,
                     "timeout-min=i" => \$timeout_min,
                     "filter-list-file=s" => \$filter_list_file,
                     "test-list-file=s" => \$test_list_file,
                     "unit-test-list-file=s" => \$unit_test_list_file,
                     "testname=s" => \$testname,
                     "dvs" => \$dvs,
                     "openmp" => \$openmp,
                     "remote_server=s" => \$remote_server,
                     "remote_android" => \$remote_android,
                     "remote_path=s" => \$remote_path,
                    );

# Generate gold output files (set to 1 manually)
my $generate_gold = 0;

my $pwd = getcwd();
my $binpath_root = abs_path ("${pwd}/..");

if ($arch eq "ARMv7") {
      if ($abi eq "") {
          $abi = "_gnueabi";  #Use default abi for arm if not specified
      }
      else {
          $abi = "_${abi}";
      }
}
elsif ($arch eq "aarch64") { 
    $abi = "_${abi}"; 
} 
else {
    $abi = "";                #Ignore abi for architectures other than arm
}

if ($remote_server || $remote_android) {
    $remote = 1;
    die "Only one of -remote_server or -remote_android can be specified on the command-line" if $remote_server && $remote_android;

    remote_check();
    if ((${remote_path} ne "") && (${remote_path} ne "/")) {
        remote_shell("rm -rf ${remote_path}");
        remote_shell("mkdir -p ${remote_path}");
    }
}

my $uname = "";
$uname = $arch;
chomp($uname);

printf ("DEBUG binpath_root=%s;\n",$binpath_root);
printf ("DEBUG uname=%s;\n",$uname);
printf ("DEBUG os=%s;\n",$os);
printf ("DEBUG substr($os,0,6)=%s;\n",substr($os,0,6));

printf ("DEBUG after Cygwin detection\n");
printf ("DEBUG uname=%s;\n",$uname);
printf ("DEBUG os=%s;\n",$os);

printf ("DEBUG binpath_root=%s;\n",$binpath_root);
my $binpath = "${binpath_root}/bin/${uname}_${os}${abi}_${build}";
printf ("DEBUG binpath=%s;\n",$binpath);

if ($remote) {
    if ($remote_server) {
        printf ("DEBUG remote_server=%s;\n",$remote_server);
    }
    printf ("DEBUG remote_path=%s;\n",$remote_path);
}

if ($valgrind_enable) {
    $tool_checker = "valgrind";
}
elsif ($cudamemcheck_enable){
    $tool_checker = $binpath . "/cuda-memcheck";
}

my %filterList;

sub remote_check {
    if ($remote_android) {
        system("adb version") && die qq(error initializing adb server, or adb not installed);
    } else {
        system("ssh -V > /dev/null 2> /dev/null") && die qq(ssh not installed properly);
        system("ssh $remote_server pwd > /dev/null") && die qq(ssh to ${remote_server} not working);
    }
}
sub remote_push {
    my ($s, $t) = @_;

    print ("remote push $s $t\n");
    if ($remote_android) {
        system("adb push ${s} ${t}") && die qq(Problem pushing $s to $t on android device);
    } else {
        system("scp -q ${s} $remote_server:${t}") && die qq(Problem pushing $s to $t on server $remote_server);
    }
}

sub remote_pull {
    my ($s, $t) = @_;

    print ("remote pull $s $t\n");
    if ($remote_android) {
        system("adb pull ${s} ${t}") && die qq(Problem pulling $t from $s on android device);
    } else {
        system("scp -q $remote_server:${s} ${t}") && die qq(Problem pulling $t from $s on server $remote_server);
    }
}

sub remote_shell {
    my $cmd = shift;
    my $ret = 0;

    print ("remote shell \"$cmd\"\n");
    if ($remote_android) {
        my $tmp = File::Temp->new( TEMPLATE => 'thrust_XXXXX' );
        my $adbtmp = "/data/thrust_adb_tmp_" . sprintf("%05u", rand(100000));
        $ret = (
                system("adb shell \"$cmd; echo $? > $adbtmp\"")
                || remote_pull("$adbtmp", "$tmp")
                || system("adb shell \"rm $adbtmp\"")
               );

        if ($ret == 0) {
            open(RETFILE, $tmp);
            $ret = <RETFILE>;
            close (RETFILE);

            chomp $ret;
            if ($ret =~ /^(\d+)/) { # Make sure to interpret cases with no return code as failure
                $ret = int($1);
            } else {
                $ret = 1;
            }
        } else {
            die ("remote shell and/or return code failed!")
        }
    } else {
        $ret = system("ssh $remote_server $cmd");
    }

    return $ret;
}

sub isFiltered {
    my $cmd = shift;

    return 0 if not defined $filter_list_file;

    if (not %filterList) {
        my $fin;
        open $fin, "<$filter_list_file" or die qq(open failed on $fin);
        foreach my $line (<$fin>) {
            chomp $line;
            $filterList{$line} = 1;
        }
        close $fin;
    }

    return $filterList{$cmd};
}

#sub getTest {
#    my ($t, $el, $utl) = @_;
#
#    $t =~ s/\s+$//;
#    if (grep(/^$t$/, @examplelist_all)) {
#        push (@$el, $t);
#    } elsif ($t =~ m/\w/) {
#        push (@$utl, $t);
#    }
#}

sub getTestList {
    my ($f, $el, $utl) = @_;
    my $fin;

    die qq(no test list file defined) if not defined $f;
    open $fin, "<$f" or die qq(open failed on $f: $!);
    foreach my $line (<$fin>) {
        getTest($line, \@$el, \@$utl);
    }
    close $fin;
}

# deprecated; marked for deletion
sub xgetUnitTestList {
    my ($f) = @_;
    my $fin;
    my @utl;

    my $tester = "thrust_test";
    if ($openmp) {
        $tester = $tester . "_OMP";
    }

    die qq(no test list file defined) if not defined $f;
    open $fin, "<$f" or die qq(open failed on $f: $!);
    foreach my $line (<$fin>) {
        $line =~ s/\s+$//;
        # Put $line in quotes to avoid <> problems
        push (@utl, "thrust_test \"$line\"");
    }
    close $fin;
    return @utl;
}

sub clear_libpath {
    if ($os eq "Darwin") {
        $ENV{'DYLD_LIBRARY_PATH'} = "";
        printf ("DYLD_LIBRARY_PATH = %s\n",$ENV{'DYLD_LIBRARY_PATH'}); 
    } elsif ($os eq "Linux") {
        $ENV{'LD_LIBRARY_PATH'} = "";
        printf ("LD_LIBRARY_PATH = %s\n",$ENV{'LD_LIBRARY_PATH'}); 
    } elsif ($os eq "win32") {
        if ($cygwin) {
            $ENV{'PATH'} = "/usr/local/bin:/usr/bin:/bin:/cygdrive/c/WINDOWS/system32";
        } else {
            $ENV{'PATH'} = "c:/Windows/system32";
        }
        printf ("PATH = %s\n",$ENV{'PATH'});
    }
}

# Wrapper for system that logs the commands so you can see what it did
sub run_cmd {
    my ($cmd) = @_;
    my  $ret = 0;
    my @executable;
    my $syst_cmd;

    print "Running $cmd\n";    

    eval {
        local $SIG{ALRM} = sub {die "alarm\n"};
        alarm (60 * $timeout_min);
        if ($tool_checker ne "") {
            $syst_cmd = $tool_checker . " " . $cmd;
        } else {
            $syst_cmd = $cmd;
        }
          
        @executable = split(' ', $syst_cmd, 2);
        if ($remote) {
            $ret = remote_shell($syst_cmd);
        } else {
            $ret = system $syst_cmd;
        }

        alarm 0;
    };
    if ($@) {
        printf "\n App timeouts : killing $executable[0]\n";        
        system ("killall ".$executable[0]);
        return 1;
    }
    
    if ($ret != 0) {
        my $signals  = $ret & 127;
        my $app_exit = $ret >> 8;
        my $dumped_core = $ret & 0x80;
        if (($app_exit != 0) && ($app_exit != 0)) {
            printf "\n App exits with status $app_exit\n";
        }
        if ($signals != 0) {
            printf "\n App received signal $signals\n";
        }  
        if ($dumped_core != 0) {
            printf "\n App generated a core dump\n";
        }                    
    }
    return $ret;
}

# Temporarily Disabling test -- http://nvbugs/1552018
# The custom_temporary_allocation example only works with gcc versions 4.4 or higher
#if (($os eq "win32") || (-e "${binpath}/custom_temporary_allocation")) {
#    push(@examplelist_all, "custom_temporary_allocation");
#}

#if (defined $testname) {
#    getTest($testname, \@examplelist, \@unittestlist);
#} elsif (defined $test_list_file) {
#    getTestList($test_list_file, \@examplelist, \@unittestlist);
#} else {
#    @examplelist = @examplelist_all;  # run all examples if -testname or 
#    @unittestlist = getUnitTestList($unit_test_list_file);
#}

sub print_time {
    my ($sec, $min, $hour, $mday, $mon, $year, $wday, $yday, $isdst) =
        localtime(time);
    printf ("current time: %02d:%02d:%02d\n", $hour, $min, $sec);
}

sub get_file {
    my ($filename, $strip) = @_;
    my $failure_output_limit=1000;
    my @stdout_output;
    my $line;

    open(OUTFILE, $filename);
    while(<OUTFILE>) {
        if (@stdout_output < $failure_output_limit) {
            $line = $_;
            if ($strip) {
                # remove all trailing whitespace
                # required for cross-platform gold file comparisons
                $line =~ s/\s+$//;
            }
            push @stdout_output, $line;
        }
    }
    close(OUTFILE);
    return @stdout_output;
}

sub compare_arrays {
    my ($first, $second) = @_;
    no warnings;  # silence spurious -w undef complaints
    return 0 unless @$first == @$second;
    for (my $i = 0; $i < @$first; $i++) {
        return 0 if $first->[$i] ne $second->[$i];
    }
    return 1;
}  

my $passed = 0;
my $failed = 0;

sub is_skip_gold_verify {
    my $test = shift;
    foreach my $skip (@skip_gold_verify_list)
    {
        if ($test eq $skip)
        {
            return 1;
        }
    }
    return 0;
}

sub run_examples {
    my $outputlog = "stderr.output";
    my $test;

    # git list of tests in binary folder
    my $dir = cwd();
    chdir $binpath;
    my @examplelist;
    if ($os eq "win32")
    {
        @examplelist = glob('thrust.example.*.exe');
    } else {
        @examplelist = glob('thrust.example.*');
    }

    chdir $dir;

    foreach $test (@examplelist)
    {
        my $test_exe = $test;
        if ($os eq "win32")
        {
            $test =~ s/\.exe//g;
        }
        # Check its not filtered via the filter file
        next if isFiltered($test);
        # Check the test actually exists
        next unless (-e "${binpath}/${test_exe}");
        print_time;

        my $ret;
        my $cmd;

        if ($remote) {
            remote_push("${binpath}/${test_exe}", "${remote_path}/${test}");
            if ($remote_android) {
                $cmd = "${remote_path}/${test_exe} > ${remote_path}/${test}.output 2> ${remote_path}/${test}.${outputlog}";
            } else {
                $cmd = "\"${remote_path}/${test_exe} > ${remote_path}/${test}.output 2> ${remote_path}/${test}.${outputlog}\"";
            }
        } else {
            $cmd = "${binpath}/${test_exe} > internal/test/${test}.output 2>> internal/test/examples.$outputlog";
        }
        open(FILE, ">>internal/test/examples.$outputlog");
        print FILE "CMD: $cmd\n";
        close(FILE);
        print "&&&& RUNNING $test\n";
        $ret = run_cmd $cmd;
        if ($remote) {
            remote_pull("${remote_path}/${test}.output", "internal/test/${test}.output");
            remote_pull("${remote_path}/${test}.${outputlog}", "internal/test/${test}.${outputlog}");
            system("cat internal/test/${test}.${outputlog} >> internal/test/examples.${outputlog}");
        }
        my @output = get_file("internal/test/${test}.output", 0);
        print @output;
        if ($ret != 0) {
            print "&&&& FAILED $test\n";
            $failed = $failed + 1;
        } elsif (is_skip_gold_verify($test)) {
            print " >>>> skip gold comparison\n";
            print "&&&& PASSED $test\n";
            $passed = $passed + 1;
        } else {
            if (-f "internal/test/${test}.gold") {
                # check output against gold file
                my @stripped_output = get_file("internal/test/${test}.output", 1);
                my @gold_output = get_file("internal/test/${test}.gold", 1);
                if (compare_arrays(\@gold_output, \@stripped_output)) {
                    print "&&&& PASSED $test\n";
                    $passed = $passed + 1;
                } else {
                    print "!!!! Bad gold comparison\n";
                    print "&&&& FAILED $test\n";
                    $failed = $failed + 1;
                }
            } else {
                print "^^^^ no gold comparison\n";
                print "&&&& PASSED $test\n";
                $passed = $passed + 1;
            }
            if ($generate_gold) {
                open(FILE, ">internal/test/${test}.gold");
                print FILE @output;
                close(FILE);
            }
        }
    }
}

# deprecated sub; marked for deletion
sub xrun_unit_tests {
    my $outputlog = "stderr.output";
    my $test_cmd;
    my $test;
    my $tester;
    my $cmd;
    my $copied_tester = 0;

    foreach $test_cmd (@unittestlist)
    {
        ($tester, $test) = split(/ /, $test_cmd);
        $test =~ s/\"//g;

        if ($remote && -f "${binpath}/${tester}" && ($copied_tester == 0)) {
            remote_push("${binpath}/${tester}", "${remote_path}/${tester}");
            $copied_tester = 1;
        }

        print_time;
        next if isFiltered("$tester \"$test\"");
        my $ret;

        print "&&&& RUNNING $tester \"$test\"\n";
        if ($remote) {
                if ($remote_android) {
                    $cmd = "${remote_path}/${tester} \\\"${test}\\\"";
                } else {
                    $cmd = "${remote_path}/${tester} \"\\\"${test}\\\"\"";
                }
        } else {
            $cmd = "${binpath}/${tester} \"${test}\"";
        }
        $ret = run_cmd $cmd;
        if ($ret != 0) {
            print "&&&& FAILED $tester \"$test\"\n";
            $failed = $failed + 1;
        } else {
            print "&&&& PASSED $tester \"$test\"\n";
            $passed = $passed + 1;
        }
    }
}
sub run_unit_tests {
    my $outputlog = "stderr.output";
    my $test;

    # git list of tests in binary folder
    my $dir = cwd();
    chdir $binpath;
    my @unittestlist;
    if ($os eq "win32")
    {
        @unittestlist = glob('thrust.test.*.exe');
    } else {
        @unittestlist = glob('thrust.test.*');
    }
    chdir $dir;

    foreach $test (@unittestlist)
    {
        my $test_exe = $test;
        if ($os eq "win32")
        {
            $test =~ s/\.exe//g;
        }
        # Check its not filtered via the filter file
        next if isFiltered($test);
        # Check the test actually exists
        next unless (-e "${binpath}/${test_exe}");
        print_time;

        my $ret;
        my $cmd;

        if ($remote) {
            remote_push("${binpath}/${test_exe}", "${remote_path}/${test}");
            if ($remote_android) {
                $cmd = "${remote_path}/${test_exe} --verbose --device=0 > ${remote_path}/${test}.output 2> ${remote_path}/${test}.${outputlog}";
            } else {
                $cmd = "\"${remote_path}/${test_exe} --verbose --device=0 > ${remote_path}/${test}.output 2> ${remote_path}/${test}.${outputlog}\"";
            }
        } else {
            $cmd = "${binpath}/${test_exe} --verbose --device=0 > internal/test/${test}.output 2>> internal/test/testing.$outputlog";
        }
        open(FILE, ">>internal/test/testing.$outputlog");
        print FILE "CMD: $cmd\n";
        close(FILE);
        print "&&&& RUNNING $test\n";
        $ret = run_cmd $cmd;
        if ($remote) {
            remote_pull("${remote_path}/${test}.output", "internal/test/${test}.output");
            remote_pull("${remote_path}/${test}.${outputlog}", "internal/test/${test}.${outputlog}");
            system("cat internal/test/${test}.${outputlog} >> internal/test/${outputlog}");
        }
        my @output = get_file("internal/test/${test}.output", 0);

        my $fail = 0;
        my $known_fail = 0;
        my $pass = 0;
        foreach my $line (@output)
        {
            my @split_line = split(/ /,$line);
            my $name = @split_line[-1];
            chomp $name;
            if (index($line, "[PASS]") != -1)
            {
                $pass = 1;
                $passed = $passed + 1;
                print "&&&& PASSED ${test}--${name} \n";
            }
            elsif (index($line, "[KNOWN FAILURE]") != -1)
            {
                $known_fail = 1;
                $passed = $passed + 1;
                print "&&&& PASSED ${test}--${name} with [KNOWN FAILURE]\n";
            }
            elsif (index($line, "[FAILURE]") != -1)
            {
                $fail = 1;
                $failed = $failed + 1;
                print "&&&& FAILED ${test}--${name} \n";
            }
        }
        if ($ret == 0) {
            if ($fail == 1)
            {
                $failed = $failed + 1;
                print "&&&& FAILED $test : \$ret = 0, while \$fail = 1 -- Undefined behaviour.\n"
            } elsif ($pass == 0 && $known_fail == 0) {
                $failed = $failed + 1;
                print "&&&& FAILED $test : \$ret = 0, while both \$pass & \$fail = 0 -- Are you sure you ran correct test?\n"
            }
        }  elsif ($fail == 0) {
            $failed = $failed + 1;
            print "&&&& FAILED $test : \$ret = 1, while \$fail = 0 -- Test crash?\n"
        }
    }
}

sub dvs_summary {

  if ( $dvs ) {
     my $dvs_score;
     my $denominator = $passed + $failed;
     if ($denominator == 0) {
        $dvs_score = 0;
     }
     else {
        $dvs_score = 100*($passed/($passed+$failed));
     }
     print "\n";
     print "RESULT\n";
     print "Passes         : $passed\n";
     print "Failures       : $failed\n";
     printf "CUDA DVS BASIC SANITY SCORE: %.1f\n",$dvs_score;
  }

}

sub current_time()
{
   my ($sec, $min, $hour, $mday, $mon, $year, $wday, $yday, $isdst) = localtime(time);
   $year += 1900;
   $mon += 1;
   return sprintf ("%04d-%02d-%02d %02d:%02d:%02d", $year, $mon, $mday, $hour, $min, $sec);
}

my $START_TIME = current_time();

print_time();
clear_libpath();
run_examples();
run_unit_tests();

my $STOP_TIME = current_time();

print "%*%*%*%* PASS3D $passed %*%*%*%*\n";
print "%*%*%*%* FA!L3D $failed %*%*%*%*\n";

print "\n";
print "Start time : $START_TIME\n";
print "Stop time  : $STOP_TIME\n";

dvs_summary();
