#! /usr/bin/perl

###############################################################################
# Copyright (c) 2018 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

use strict;
use warnings;

print `perl --version`;

use Getopt::Long;
use Cwd;
use Cwd "abs_path";
use Config; # For signal names and numbers.
use File::Temp;
use POSIX "strftime";

my $have_time_hi_res = 0;

if (eval { require Time::HiRes })
{
  printf("#### CONFIG timestamp `gettimeofday`\n");

  import Time::HiRes "gettimeofday";

  $have_time_hi_res = 1;
} else {
  printf("#### CONFIG timestamp `time`\n");
}

sub timestamp()
{
  if ($have_time_hi_res) {
    return gettimeofday();
  } else {
    return time();
  }
}

my %CmdLineOption;
my $retVal;
my $arch                = "";
my $build               = "release";
my $bin_path;
my $filecheck_path;
my $filecheck_data_path = "internal/test";
my $filter_list_file    = undef;
my $testname            = undef;
my $valgrind_enable     = 0;
my $cudamemcheck_enable = 0;
my $tool_checker        = "";
my $timeout_min         = 15;
my $os                  = "";
my $cygwin              = "";
my $openmp              = 0;
my $config              = "";
my $abi                 = "";
my $remote              = "";
my $remote_server       = "";
my $remote_android      = "";
my $remote_path         = "/data/thrust_testing";

# https://stackoverflow.com/questions/29862178/name-of-signal-number-2
my @sig_names;
@sig_names[ split ' ', $Config{sig_num} ] = split ' ', $Config{sig_name};
my %sig_nums;
@sig_nums{ split ' ', $Config{sig_name} } = split ' ', $Config{sig_num};

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
      (lc($ENV{PROCESSOR_ARCHITEW6432}) eq "amd64")) {
    $arch = "x86_64";
  } else {
    $arch = "i686";
  }
} else {
  $arch = `uname -m`;
  chomp($arch);
}

sub usage()
{
  printf("Usage: thrust_nightly.pl <options>\n");
  printf("Options:\n");
  printf("  -help                         : Print help message\n");
  printf("  -forcearch <arch>             : i686|x86_64|ARMv7|aarch64 (default: $arch)\n");
  printf("  -forceabi <abi>               : Specify abi to be used for arm (gnueabi|gnueabihf)\n");
  printf("  -forceos <os>                 : win32|Linux|Darwin (default: $os)\n");
  printf("  -build <release|debug>        : (default: debug)\n");
  printf("  -bin-path <path>              : Specify location of test binaries\n");
  printf("  -filecheck-path <path>        : Specify location of filecheck binary\n");
  printf("  -filecheck-data-path <path>   : Specify location of filecheck data (default: $filecheck_data_path)\n");
  printf("  -timeout-min <min>            : timeout in minutes for each individual test\n");
  printf("  -filter-list-file <file>      : path to filter file which contains one invocation per line\n");
  printf("  -openmp                       : test OpenMP implementation\n");
  printf("  -remote-server <server>       : test on remote target (uses ssh)\n");
  printf("  -remote-android               : test on remote android target (uses adb)\n");
  printf("  -remote-path                  : path on remote target to copy test files (default: $remote_path)\n");
}

$retVal = GetOptions(\%CmdLineOption,
                     'help'     => sub { usage() and exit 0 },
                     "forcearch=s" => \$arch,
                     "forceabi=s" => \$abi,
                     "forceos=s" => \$os,
                     "build=s" => \$build,
                     "bin-path=s" => \$bin_path,
                     "filecheck-path=s" => \$filecheck_path,
                     "filecheck-data-path=s" => \$filecheck_data_path,
                     "timeout-min=i" => \$timeout_min,
                     "filter-list-file=s" => \$filter_list_file,
                     "openmp" => \$openmp,
                     "remote-server=s" => \$remote_server,
                     "remote-android" => \$remote_android,
                     "remote-path=s" => \$remote_path,
                    );

my $pwd = getcwd();
my $bin_path_root = abs_path ("${pwd}/..");

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

if (not $bin_path) {
    $bin_path = "${bin_path_root}/bin/${uname}_${os}${abi}_${build}";
}

if (not $filecheck_path) {
    $filecheck_path = "${bin_path}/nvvm/tools";
}

if ($valgrind_enable) {
    $tool_checker = "valgrind";
}
elsif ($cudamemcheck_enable){
    $tool_checker = $bin_path . "/cuda-memcheck";
}

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

    printf("#### REMOTE_PUSH $s $t\n");
    if ($remote_android) {
        system("adb push ${s} ${t}") && die qq(Problem pushing $s to $t on android device);
    } else {
        system("scp -q ${s} $remote_server:${t}") && die qq(Problem pushing $s to $t on server $remote_server);
    }
}

sub remote_pull {
    my ($s, $t) = @_;

    printf("#### REMOTE_PULL $s $t\n");
    if ($remote_android) {
        system("adb pull ${s} ${t}") && die qq(Problem pulling $t from $s on android device);
    } else {
        system("scp -q $remote_server:${s} ${t}") && die qq(Problem pulling $t from $s on server $remote_server);
    }
}

sub remote_shell {
    my $cmd = shift;
    my $ret = 0;

    printf("#### REMOTE_SHELL `$cmd`\n");
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

my %filter_list;

sub is_filtered {
    my $cmd = shift;

    return 0 if not defined $filter_list_file;

    if (not %filter_list) {
        my $fin;
        open $fin, "<$filter_list_file" or die qq(open failed on $fin);
        foreach my $line (<$fin>) {
            chomp $line;
            $filter_list{$line} = 1;
        }
        close $fin;
    }

    return $filter_list{$cmd};
}

sub clear_libpath {
    if ($os eq "Darwin") {
        $ENV{'DYLD_LIBRARY_PATH'} = "";
        printf("#### CONFIG DYLD_LIBRARY_PATH `%s`\n", $ENV{'DYLD_LIBRARY_PATH'});
    } elsif ($os eq "Linux") {
        # When running under `nvidia-docker`, clearing `LD_LIBRARY_PATH` breaks
        # the build. Currently, there's no good way to determine if we're
        # running under `nvidia-docker`. The best idea I could come up with was
        # to match against the `LD_LIBRARY_PATH` that `nvidia-docker` sets.
        # https://nvbugswb.nvidia.com/NvBugs5/SWBug.aspx?bugid=2003238
        if (defined($ENV{'LD_LIBRARY_PATH'})) {
            if ($ENV{'LD_LIBRARY_PATH'} ne "/usr/local/nvidia/lib:/usr/local/nvidia/lib64") {
                $ENV{'LD_LIBRARY_PATH'} = "";
                printf("#### CONFIG LD_LIBRARY_PATH `%s`\n", $ENV{'LD_LIBRARY_PATH'});
            }
        }
    } elsif ($os eq "win32") {
        if ($cygwin) {
            $ENV{'PATH'} = "/usr/local/bin:/usr/bin:/bin:/cygdrive/c/WINDOWS/system32";
        } else {
            $ENV{'PATH'} = "c:/Windows/system32";
        }
        printf("#### CONFIG PATH `%s`\n", $ENV{'PATH'});
    }
}

sub process_return_code {
    my ($name, $ret, $msg) = @_;

    if ($ret != 0) {
        my $signal  = $ret & 127;
        my $app_exit = $ret >> 8;
        my $dumped_core = $ret & 0x80;
        if (($app_exit != 0) && ($app_exit != 0)) {
            if ($msg ne "") {
                printf("#### ERROR $name exited with return value $app_exit. $msg\n");
            } else {
                printf("#### ERROR $name exited with return value $app_exit.\n");
            }
        }
        if ($signal != 0) {
            if ($msg ne "") {
                printf("#### ERROR $name received signal SIG$sig_names[$signal] ($signal). $msg\n");
            } else {
                printf("#### ERROR $name received signal SIG$sig_names[$signal] ($signal).\n");
            }
            if ($sig_nums{'INT'} eq $signal) {
                die("Terminating testing due to SIGINT.");
            }
        }
        if ($dumped_core != 0) {
            if ($msg ne "") {
                printf("#### ERROR $name generated a core dump. $msg\n");
            } else {
                printf("#### ERROR $name generated a core dump.\n");
            }
        }
    }
}

# Wrapper for system that logs the commands so you can see what it did
sub run_cmd {
    my ($cmd) = @_;
    my  $ret = 0;
    my @executable;
    my $syst_cmd;

    my $start = timestamp();
    eval {
        local $SIG{ALRM} = sub { die("Command timed out (received SIGALRM).\n") };
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
    my $elapsed = timestamp() - $start;

    if ($@) {
        printf("\n#### ERROR Command timeout reached, killing $executable[0].\n");
        system("killall ".$executable[0]);
        return ($sig_nums{'KILL'}, $elapsed);
    }

    return ($ret, $elapsed);
}

sub current_time
{
   return strftime("%x %X %Z", localtime());
}

sub get_file {
    my ($filename) = @_;

    open(my $handle, '<', $filename);
    my @output = <$handle>;
    close($handle);

    return @output;
}

my $failures = 0;
my $known_failures = 0;
my $errors = 0;
my $passes = 0;

sub run_examples {
    # Get list of tests in binary folder.
    my $dir = cwd();
    chdir $bin_path;
    my @examplelist;
    if ($os eq "win32")
    {
        @examplelist = glob('thrust.example.*.exe');
    } else {
        @examplelist = glob('thrust.example.*');
    }

    chdir $dir;

    my $test;
    foreach $test (@examplelist)
    {
        my $test_exe = $test;
        if ($os eq "win32")
        {
            $test =~ s/\.exe//g;
        }
        # Check its not filtered via the filter file
        next if is_filtered($test);
        # Check the test actually exists
        next unless (-e "${bin_path}/${test_exe}");

        my $cmd;

        if ($remote) {
            remote_push("${bin_path}/${test_exe}", "${remote_path}/${test}");
            if ($remote_android) {
                $cmd = "${remote_path}/${test_exe} --verbose > ${remote_path}/${test}.output 2>&1";
            } else {
                $cmd = "\"${remote_path}/${test_exe} --verbose > ${remote_path}/${test}.output 2>&1\"";
            }
        } else {
            $cmd = "${bin_path}/${test_exe} --verbose > ${test}.output 2>&1";
        }

        printf("&&&& RUNNING $test\n");
        printf("#### CURRENT_TIME " . current_time() . "\n");

        my ($ret, $elapsed) = run_cmd $cmd;
        if ($remote) {
            remote_pull("${remote_path}/${test}.output", "${test}.output");
        }
        my @output = get_file("${test}.output");
        printf("********************************************************************************\n");
        print @output;
        printf("********************************************************************************\n");
        if ($ret != 0) {
            process_return_code($test, $ret, "Example crash?");
            printf("&&&& FAILED $test\n");
            printf("#### WALLTIME $test %.2f [s]\n", $elapsed);
            $errors = $errors + 1;
        } else {
            printf("&&&& PASSED $test\n");
            printf("#### WALLTIME $test %.2f [s]\n", $elapsed);
            $passes = $passes + 1;

            # Check output with LLVM FileCheck.

            my $filecheck = "${filecheck_path}/FileCheck --input-file ${test}.output ${filecheck_data_path}/${test}.filecheck > ${test}.filecheck.output 2>&1";

            printf("&&&& RUNNING FileCheck $test\n");

            if (-f "${filecheck_data_path}/${test}.filecheck") {
                # If the filecheck file is empty, don't use filecheck, just
                # check if the output file is also empty.
                if (-z "${filecheck_data_path}/${test}.filecheck") {
                    if (-z "${test}.output") {
                        printf("&&&& PASSED FileCheck $test\n");
                        $passes = $passes + 1;
                    } else {
                        printf("#### Output received but not expected.\n");
                        printf("&&&& FAILED FileCheck $test\n");
                        $failures = $failures + 1;
                    }
                } else {
                    if (system($filecheck) == 0) {
                        printf("&&&& PASSED FileCheck $test\n");
                        $passes = $passes + 1;
                    } else {
                        my @filecheckoutput = get_file("${test}.filecheck.output");
                        printf("********************************************************************************\n");
                        print @filecheckoutput;
                        printf("********************************************************************************\n");
                        printf("&&&& FAILED FileCheck $test\n");
                        $failures = $failures + 1;
                    }
                }
            } else {
                printf("#### ERROR $test has no FileCheck comparison.\n");
                printf("&&&& FAILED FileCheck $test\n");
                $errors = $errors + 1;
            }
        }
        printf("\n");
    }
}

sub run_unit_tests {
    # Get list of tests in binary folder.
    my $dir = cwd();
    chdir $bin_path;
    my @unittestlist;
    if ($os eq "win32")
    {
        @unittestlist = glob('thrust.test.*.exe');
    } else {
        @unittestlist = glob('thrust.test.*');
    }
    chdir $dir;

    my $test;
    foreach $test (@unittestlist)
    {
        my $test_exe = $test;
        if ($os eq "win32")
        {
            $test =~ s/\.exe//g;
        }
        # Check its not filtered via the filter file
        next if is_filtered($test);
        # Check the test actually exists
        next unless (-e "${bin_path}/${test_exe}");

        my $cmd;

        if ($remote) {
            remote_push("${bin_path}/${test_exe}", "${remote_path}/${test}");
            if ($remote_android) {
                $cmd = "${remote_path}/${test_exe} --verbose > ${remote_path}/${test}.output 2>&1";
            } else {
                $cmd = "\"${remote_path}/${test_exe} --verbose > ${remote_path}/${test}.output 2>&1\"";
            }
        } else {
            $cmd = "${bin_path}/${test_exe} --verbose > ${test}.output 2>&1";
        }

        printf("&&&& RUNNING $test\n");
        printf("#### CURRENT_TIME " . current_time() . "\n");

        my ($ret, $elapsed) = run_cmd $cmd;
        if ($remote) {
            remote_pull("${remote_path}/${test}.output", "${test}.output");
        }
        my @output = get_file("${test}.output");
        printf("********************************************************************************\n");
        print @output;
        printf("********************************************************************************\n");
        my $fail = 0;
        my $known_fail = 0;
        my $error = 0;
        my $pass = 0;
        my $found_totals = 0;
        foreach my $line (@output)
        {
            if (($fail, $known_fail, $error, $pass) = $line =~ /Totals: ([0-9]+) failures, ([0-9]+) known failures, ([0-9]+) errors, and ([0-9]+) passes[.]/igs) {
              $found_totals = 1;
              $failures = $failures + $fail;
              $known_failures = $known_failures + $known_fail;
              $errors = $errors + $error;
              $passes = $passes + $pass;
              last;
            } else {
              $fail = 0;
              $known_fail = 0;
              $error = 0;
              $pass = 0;
            }
        }
        if ($ret == 0) {
            if ($found_totals == 0) {
                $errors = $errors + 1;
                printf("#### ERROR $test returned 0 and no summary line was found. Invalid test?\n");
                printf("&&&& FAILED $test\n");
                printf("#### WALLTIME $test %.2f [s]\n", $elapsed);
            }
            else {
                if ($fail != 0 or $error != 0) {
                    $errors = $errors + 1;
                    printf("#### ERROR $test returned 0 and had failures or errors. Test driver error?\n");
                    printf("&&&& FAILED $test\n");
                    printf("#### WALLTIME $test %.2f [s]\n", $elapsed);
                } elsif ($known_fail == 0 and $pass == 0) {
                    $errors = $errors + 1;
                    printf("#### ERROR $test returned 0 and had no failures, known failures, errors or passes. Invalid test?\n");
                    printf("&&&& FAILED $test\n");
                    printf("#### WALLTIME $test %.2f [s]\n", $elapsed);
                } else {
                    printf("&&&& PASSED $test\n");
                    printf("#### WALLTIME $test %.2f [s]\n", $elapsed);

                    # Check output with LLVM FileCheck if the test has a FileCheck input.

                    my $filecheck = "${filecheck_path}/FileCheck --input-file ${test}.output ${filecheck_data_path}/${test}.filecheck > ${test}.filecheck.output 2>&1";

                    if (-f "${filecheck_data_path}/${test}.filecheck") {
                        printf("&&&& RUNNING FileCheck $test\n");

                        # If the filecheck file is empty, don't use filecheck,
                        # just check if the output file is also empty.
                        if (! -z "${filecheck_data_path}/${test}.filecheck") {
                            if (-z "${test}.output") {
                                printf("&&&& PASSED FileCheck $test\n");
                                $passes = $passes + 1;
                            } else {
                                printf("#### Output received but not expected.\n");
                                printf("&&&& FAILED FileCheck $test\n");
                                $failures = $failures + 1;
                            }
                        } else {
                            if (system($filecheck) == 0) {
                                printf("&&&& PASSED FileCheck $test\n");
                                $passes = $passes + 1;
                            } else {
                                my @filecheckoutput = get_file("${test}.filecheck.output");
                                printf("********************************************************************************\n");
                                print @filecheckoutput;
                                printf("********************************************************************************\n");
                                printf("&&&& FAILED FileCheck $test\n");
                                $failures = $failures + 1;
                            }
                        }
                    }
                }
            }
        } else {
            $errors = $errors + 1;
            process_return_code($test, $ret, "Test crash?");
            printf("&&&& FAILED $test\n");
            printf("#### WALLTIME $test %.2f [s]\n", $elapsed);
        }
        printf("\n");
    }
}

sub dvs_summary {
    my $dvs_score = 0;
    my $denominator = $failures + $known_failures + $errors + $passes;
    if ($denominator == 0) {
       $dvs_score = 0;
    }
    else {
       $dvs_score = 100 * (($passes + $known_failures) / $denominator);
    }

    printf("\n");

    printf("%*%*%*%* FA!LUR3S       $failures\n");
    printf("%*%*%*%* KN0WN FA!LUR3S $known_failures\n");
    printf("%*%*%*%* 3RR0RS         $errors\n");
    printf("%*%*%*%* PASS3S         $passes\n");

    printf("\n");

    printf("CUDA DVS BASIC SANITY SCORE : %.1f\n", $dvs_score);

    if ($failures + $errors > 0) {
        exit(1);
    }
}

###############################################################################

printf("#### CONFIG os `%s`\n", $os);
printf("#### CONFIG bin_path `%s`\n", $bin_path);
  
printf("#### CONFIG have_time_hi_res `$have_time_hi_res`\n");

if ($remote) {
  if ($remote_server) {
    printf("#### CONFIG remote_server `%s`\n", $remote_server);
  }
  printf("#### CONFIG remote_path `%s`\n", $remote_path);
}

printf("\n");

my $START_TIME = current_time();

clear_libpath();
run_examples();
run_unit_tests();

my $STOP_TIME = current_time();

printf("\n");

printf("#### START_TIME $START_TIME\n");
printf("#### STOP_TIME $STOP_TIME\n");

dvs_summary();

