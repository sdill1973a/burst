# det/Makefile — detrc: the shared library for Python (x86 host) and the cross-arch harness (x86 + aarch64 under qemu)
CFLAGS   ?= -O2 -ffp-contract=off -Wall -Wextra -std=c99
NDK      ?= $(ANDROID_HOME)/ndk/27.2.12479018/toolchains/llvm/prebuilt/linux-x86_64/bin
ARMCC    ?= $(NDK)/aarch64-linux-android31-clang
QEMU     ?= /usr/bin/qemu-aarch64
B        ?= build

all: $(B)/libdetrc.so $(B)/detrc_test_x86 $(B)/detrc_test_aarch64 $(B)/detrc_ctl_x86 $(B)/detrc_ctl_aarch64

$(B):
	mkdir -p $(B)

$(B)/libdetrc.so: detrc.c detrc.h detrc_exptab.h | $(B)
	$(CC) $(CFLAGS) -fPIC -shared detrc.c -o $@

$(B)/detrc_test_x86: detrc.c detrc_test.c detrc.h detrc_exptab.h | $(B)
	$(CC) $(CFLAGS) detrc.c detrc_test.c -o $@

$(B)/detrc_test_aarch64: detrc.c detrc_test.c detrc.h detrc_exptab.h | $(B)
	$(ARMCC) $(CFLAGS) -static detrc.c detrc_test.c -o $@

# negative control: a float softmax (libm expf, float sums) with FMA contraction + fast-math ON for the arm build.
# Measured 10-24: with contraction OFF, glibc and bionic expf AGREED on every synthetic row (that control could not
# fail); with -O3 -ffast-math -ffp-contract=fast on arm64 the freq tables and bytes DIFFER (byte 24 of the stream).
# The integer detrc build is identical under the same flags. `make identity` asserts the control's divergence.
$(B)/detrc_ctl_x86: detrc.c detrc_test.c | $(B)
	$(CC) $(CFLAGS) -DCONTROL_FLOAT detrc.c detrc_test.c -o $@ -lm

$(B)/detrc_ctl_aarch64: detrc.c detrc_test.c | $(B)
	$(ARMCC) -O3 -ffast-math -ffp-contract=fast -Wall -std=c99 -DCONTROL_FLOAT -static detrc.c detrc_test.c -o $@ -lm

# identity gate: same freq tables + same bytes on both chips; the control build must differ
identity: $(B)/detrc_test_x86 $(B)/detrc_test_aarch64 $(B)/detrc_ctl_x86 $(B)/detrc_ctl_aarch64
	$(B)/detrc_test_x86 synth $(B)/freq_x86.bin $(B)/rc_x86.bin
	$(QEMU) $(B)/detrc_test_aarch64 synth $(B)/freq_arm.bin $(B)/rc_arm.bin
	cmp $(B)/freq_x86.bin $(B)/freq_arm.bin && cmp $(B)/rc_x86.bin $(B)/rc_arm.bin && echo "detrc: IDENTICAL x86 vs aarch64"
	$(B)/detrc_ctl_x86 synth $(B)/cfreq_x86.bin $(B)/crc_x86.bin
	$(QEMU) $(B)/detrc_ctl_aarch64 synth $(B)/cfreq_arm.bin $(B)/crc_arm.bin
	@if cmp -s $(B)/crc_x86.bin $(B)/crc_arm.bin; then echo "CONTROL DID NOT DIFFER: the gate cannot fail"; exit 1; \
	else echo "control (float softmax, FMA on arm): DIFFERS as it must"; fi
	$(B)/detrc_test_x86 synth1ulp $(B)/freq_x86_1.bin $(B)/rc_x86_1.bin
	@if cmp -s $(B)/rc_x86.bin $(B)/rc_x86_1.bin; then echo "INPUT-CHANGE CONTROL DID NOT DIFFER"; exit 1; \
	else echo "control (one input float changed): DIFFERS as it must"; fi

clean:
	rm -rf $(B)

.PHONY: all identity clean
